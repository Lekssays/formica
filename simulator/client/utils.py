import base64
import modelUpdate_pb2
import score_pb2
import torch
import io
import requests
import json
import os
import time
import random
import websockets
import redis
import json
import numpy as np

from collections import OrderedDict
from datetime import datetime
from google.protobuf import text_format
from io import BytesIO
from tempfile import TemporaryFile

GOSHIMMER_API_ENDPOINT = os.getenv("GOSHIMMER_API_ENDPOINT") # http://0.0.0.0:8091
IPFS_API_ENDPOINT = os.getenv("IPFS_API_ENDPOINT") # http://0.0.0.0:5001
FORMICA_ENDPOINT = os.getenv("FORMICA_ENDPOINT") # "http://0.0.0.0:8090/formica"
MY_PUB_KEY = os.getenv("MY_PUB_KEY")

MODEL_UPDATE_PYTHON_PURPOSE_ID = 16
MODEL_UPDATE_GOLANG_PURPOSE_ID = 17
TRUST_PURPOSE_ID        = 21
SIMILARITY_PURPOSE_ID   = 22
ALIGNMENT_PURPOSE_ID    = 23
GRADIENTS_PURPOSE_ID    = 24
PHI_PURPOSE_ID          = 25

# Limit of Weights to Choose to Analyze and Train From
LIMIT_CHOOSE = 10

# Number of Weights to Train From
LIMIT_SELECTED = 3


def to_protobuf(modelID: str, parents: list, weights: str, model: str, pubkey: str, timestamp: int, accuracy: float):
    model_update = modelUpdate_pb2.ModelUpdate()
    model_update.modelID = modelID
    for parent in parents:
        model_update.parents.append(parent)
    model_update.weights = weights
    model_update.model = model
    model_update.accuracy = accuracy
    model_update.pubkey = pubkey
    model_update.timestamp = timestamp
    return model_update


def to_bytes(content: OrderedDict) -> bytes:
    buff = io.BytesIO()
    torch.save(content, buff)
    buff.seek(0)
    return buff.read()


def to_numpy_bytes(content) -> bytes:
    buff = TemporaryFile()
    np.save(buff, content)
    buff.seek(0)
    return buff.read()


def from_bytes_to_tensor(content: bytes) -> torch.Tensor:
    buff = io.BytesIO(content)
    loaded_content = torch.load(buff)
    return loaded_content



def send_model_update(model_update: modelUpdate_pb2.ModelUpdate):
    payload = {
        'purpose': MODEL_UPDATE_PYTHON_PURPOSE_ID,
        'data': text_format.MessageToString(model_update),
    }
    res = requests.post(FORMICA_ENDPOINT, json=payload)
    if "error" not in res.json():
        return res.json()['blockID']
    return None


def add_content_to_ipfs(content: bytes) -> str:
    url = IPFS_API_ENDPOINT + "/api/v0/add"
    files = {'file': BytesIO(content)}
    res = requests.post(url, files=files)
    return res.json()['Hash']


def get_content_from_ipfs(path: str):
    url = IPFS_API_ENDPOINT + "/api/v0/get?arg=" + path
    res = requests.post(url)
    buff = bytes(res.content)
    return buff[512:]


def get_resource_from_leveldb(key: str):
    r = redis.Redis(host='localhost', port=6379, db=0)
    value = r.get(key)
    return value


def store_resource_on_redis(key: str, content: bytes):
    r = redis.Redis(host='localhost', port=6379, db=0)
    res = r.set(key, content)
    return res


def get_model_update(block_id: str) -> modelUpdate_pb2.ModelUpdate:
    model_update_bytes = get_resource_from_leveldb(key=block_id)
    model_update = None
    if model_update_bytes is None:
        model_update, _ = parse_payload(blockID=block_id)
    else:
        model_update = text_format.Parse(model_update_bytes, modelUpdate_pb2.ModelUpdate())
    return model_update


def get_similarity():
    similarity_path = get_resource_from_leveldb(key="similarity")
    if similarity_path is None:
        return None
    similarity_path = str(similarity_path).split('"')
    similarity_bytes = get_content_from_ipfs(path=similarity_path[1])
    return np.load(BytesIO(similarity_bytes))


def get_trust():
    trust_path = get_resource_from_leveldb(key="trust")
    if trust_path is None:
        return None
    trust_path = str(trust_path).split('"')
    trust_bytes = get_content_from_ipfs(path=trust_path[1])
    return np.load(BytesIO(trust_bytes))


def get_purpose(payload: str):
    payload = base64.b64decode(payload)
    purpose = str(payload[3:5])
    return int("0x" + purpose[4:6] + purpose[8:10], 16)


def parse_payload(blockID: str):
    url = GOSHIMMER_API_ENDPOINT + "/api/block/" + blockID
    res = requests.get(url)
    payload = base64.b64decode(res.json()['payload']['content'])
    payload = payload[12:]
    payload = payload[:-4]
    purpose = get_purpose(res.json()['payload']['content'])
    parsed_payload = None
    if purpose == MODEL_UPDATE_GOLANG_PURPOSE_ID:
        payload = base64.b64decode(payload)
        parsed_payload = modelUpdate_pb2.ModelUpdate()
        parsed_payload.ParseFromString(payload)
    elif purpose == MODEL_UPDATE_PYTHON_PURPOSE_ID:
        parsed_payload = text_format.Parse(payload.decode(), modelUpdate_pb2.ModelUpdate())
    elif purpose in [TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, GRADIENTS_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]:
        payload = base64.b64decode(payload)
        parsed_payload = score_pb2.Score()
        parsed_payload.ParseFromString(payload)
    return parsed_payload, purpose


def get_weights_tensor(path: str) -> torch.Tensor:
    weights_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes_to_tensor(weights_from_ipfs)

def get_gradients(path: str) -> torch.Tensor:
    gradients_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes_to_tensor(gradients_from_ipfs)


def get_weights_ids(model_id, limit):
    weights = []
    with open(os.getenv("TMP_FOLDER") + model_id + ".dat", "r") as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            if line not in weights and len(line) > 0:
                weights.append(line)

    if limit >= len(weights):
        return weights

    weights.reverse()
    return weights[:limit]


def store_weight_id(modelID, blockID):
    f = open(os.getenv("TMP_FOLDER") + modelID + ".dat", "a")
    f.write(blockID + "\n")
    f.close()


def get_weights_to_train(model_id: str):
    weights = []
    indices = []
    parents = []
    timestamps = []

    chosen_weights_ids = get_weights_ids(model_id=model_id, limit=LIMIT_CHOOSE)

    metrics = []
    for block_id in chosen_weights_ids:
        mu = get_model_update(block_id=block_id)
        tmp = {
            'blockID': block_id,
            'timestamp': mu.timestamp,
        }
        metrics.append(tmp)

    metrics = sorted(metrics, key=lambda x: (x['timestamp']), reverse=True)

    limit = min(LIMIT_SELECTED, len(metrics))
    metrics = metrics[:limit]

    for m in metrics:
        mu = get_model_update(block_id=m['blockID'])
        idx = get_client_id(pubkey=mu.pubkey)
        if idx != int(os.getenv("MY_ID")):
            # get a tensor stored in ipfs
            w = get_weights_dict(path=mu.model)
            if len(w) == 46:
                w = get_weights_dict(path=w)
            weights.append(w)
            indices.append(idx)
            parents.append(m['blockID'])
            timestamps.append(m['timestamp'])

    if len(weights) > 0:
        c = list(zip(weights, indices, parents, timestamps))
        random.shuffle(c)
        weights, indices, parents, timestamps = zip(*c)

    return weights, indices, parents


def get_client_id(pubkey: str):
    with open('peers.json', "r") as f:
        peers = json.load(f)

    for peer in peers['peers']:
        if peer["pubkey"] == pubkey:
            return int(peer["id"])

    return None

def get_parameter(param: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[param]

def get_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def get_parameter_with_default(param: str, default:object):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config.get(param, default)

def get_dataset_metadata(dataset):
    data_dir_path = os.getenv("DATA_FOLDER")

    metadata_path = os.path.join(data_dir_path, dataset, "info.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def publish_model_update(modelID, accuracy, parents, model, weights):
    model_bytes = to_bytes(model)
    model_path = add_content_to_ipfs(content=model_bytes)

    weights_bytes = to_numpy_bytes(weights.astype('float64'))
    weights_path = add_content_to_ipfs(content=weights_bytes)

    model_update_pb = to_protobuf(
        modelID=modelID,
        parents=parents,
        weights=weights_path,
        pubkey=os.getenv("MY_PUB_KEY"),
        accuracy=accuracy,
        timestamp=int(time.time()),
        model=model_path
    )

    return send_model_update(model_update_pb)


def get_phi():
    phi_path = get_resource_from_leveldb(key="phi")
    if phi_path is None:
        return
    phi_path = str(phi_path).split('"')
    phi_bytes = get_content_from_ipfs(path=phi_path[1])
    return np.load(BytesIO(phi_bytes))


def process_message(message):
    message = json.loads(message)
    blockID = message['data']['id']
    payload, purpose = parse_payload(blockID=blockID)
    print(purpose, payload)
    payload_bytes = bytes(text_format.MessageToString(payload), encoding='utf8')
    if int(purpose) in [MODEL_UPDATE_PYTHON_PURPOSE_ID, MODEL_UPDATE_GOLANG_PURPOSE_ID]:
        if payload.pubkey != os.getenv("MY_PUB_KEY"):
            store_resource_on_redis(blockID, payload_bytes)
            store_weight_id(modelID=payload.modelID, blockID=blockID)
    elif int(purpose) in [TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]:
        if int(purpose) == TRUST_PURPOSE_ID:
            store_resource_on_redis("trust", payload_bytes)
        elif int(purpose) == SIMILARITY_PURPOSE_ID:
            store_resource_on_redis("similarity", payload_bytes)
        elif int(purpose) == ALIGNMENT_PURPOSE_ID:
            store_resource_on_redis("algnscore", payload_bytes)
        elif int(purpose) == PHI_PURPOSE_ID:
            store_resource_on_redis("phi", payload_bytes)


def get_my_latest_accuracy():
    acc_bytes = get_resource_from_leveldb(key='accuracy')
    if acc_bytes == None:
        return 0.00
    return float(str(acc_bytes.decode("utf-8")))


def store_my_latest_accuracy(accuracy: float):
    content = bytes(str(accuracy), encoding="utf-8")
    store_resource_on_redis(key="accuracy", content=content)


async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("MY_NAME") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)


def get_dishonest_peers():
    dishonest_peers = os.getenv("DISHONEST_PEERS")
    if dishonest_peers == "dishonest_peers":
        return list()
    dishonest_peers.split(",")
    return dishonest_peers

def get_model_state_dir_path():
    return os.path.join(os.getenv("TMP_FOLDER"))

def get_model_state_path(model_id):
    return os.path.join(get_model_state_dir_path(), "{}.pt".format(model_id))
