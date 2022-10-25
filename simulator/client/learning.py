import asyncio
import pickle
import os
from typing import OrderedDict
import models
from models.aggregators import Aggregator
import utils
import random
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from os.path import exists
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from dataloader import TrainDataset, TestDataset

torch.backends.cudnn.benchmark=True
torch.manual_seed(42)
np.random.seed(42)


def peer_update(local_model, train_loader, epoch=1, attack_type=None):
    print(os.getenv("MY_NAME"), "attack_type", attack_type)
    dataset = utils.get_parameter(param="dataset")
    optimizer = get_optimizer(local_model)
    adversarial_temperature = utils.get_parameter(param="adversarial_temperature")

    local_model.train()
    for _ in range(epoch):
        for batch in tqdm(train_loader):
            positive_sample, negative_sample, sample_idx = batch
            positive_sample = positive_sample.to(local_model.device)
            negative_sample = negative_sample.to(local_model.device)

            negative_score = local_model((positive_sample, negative_sample))

            negative_score = (F.softmax(negative_score * adversarial_temperature, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)

            positive_score = local_model(positive_sample, neg=False)
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # data, target = data, target
            # optimizer.zero_grad()
            # output = local_model(data)

            # # Attack can be added here depending on the dataset
            # if attack_type is not None:
            #     if dataset == "MNIST":
            #         for i, t in enumerate(target):
            #             if attack_type == 'lf':  # label flipping
            #                 if t == 1:
            #                     target[i] = torch.tensor(7)
            #             elif attack_type == 'backdoor':
            #                 target[i] = 1  # set the label
            #                 data[:, :, 27, 27] = torch.max(data)  # set the bottom right pixel to white.
            #             elif attack_type == 'untargeted':
            #                 target[i] = random.randint(0, 9)
            #             elif attack_type == "untargeted_sybil":  # untargeted with sybils
            #                 target[i] = 0
            # loss = F.nll_loss(output, target)
            # loss.backward()
            # optimizer.step()

    return loss.item(), local_model


def aggregate(peers_indices, peers_weights=[]):
    if len(peers_weights) == 0:
        return
    phi = utils.get_phi()
    if phi is None:
        return
    global_dict = peers_weights[-1].get_state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([peers_weights[idx].state_dict()[k].float()* phi[peers_indices[idx]] for idx in range(0, len(peers_indices))], 0).mean(0)
    peers_weights[-1].load_state_dict(global_dict)
    return peers_weights[-1]


def test(local_model, test_loader, attack):
    """This function test the global model on test data and returns test loss and test accuracy """
    # local_model.eval()
    test_loss = 0
    correct = 0
    dataset = utils.get_parameter(param="dataset")

    results = {
        "count": 0.0,
        "mr": 0.0,
        "mrr": 0.0,
    }

    hit_values = [1,5,10]
    for k in hit_values:
        results["hits@{}".format(k)] = 0

    for batch in test_loader:
        triplets, labels = batch
        triplets, labels = triplets.to(local_model.device), labels.to(local_model.device)
        head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        pred = local_model((triplets, None))
        b_range = torch.arange(pred.size()[0], device=local_model.device)
        target_pred = pred[b_range, tail_idx]
        pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, tail_idx] = target_pred

        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                    dim=1, descending=False)[b_range, tail_idx]

        ranks = ranks.float()
        count = torch.numel(ranks)

        results['count'] += count
        results['mr'] += torch.sum(ranks).item()
        results['mrr'] += torch.sum(1.0 / ranks).item()

        for k in hit_values:
            results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

    for k, v in results.items():
        if k != 'count':
            results[k] /= results['count']

    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data, target
    #         output = local_model(data)

    #         # sum up batch loss
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()

    #         # get the index of the max log-probability
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    #         if dataset == "MNIST":
    #             for i, t in enumerate(target):
    #                 if t == 1 and pred[i] == 7:
    #                     attack += 1

    # test_loss /= len(test_loader.dataset)
    # acc = correct / len(test_loader.dataset)

    attack = 0 # IMPORTANT: not sure how to calculate attack
    return results, attack


def load_data():
    train_x, train_y, test_loader = None, None, None
    dataset = utils.get_parameter(param="dataset")
    batch_size = utils.get_parameter(param="batch_size")

    if dataset == "MNIST":
        transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

        testdata = datasets.MNIST(os.getenv("DATA_FOLDER") + 'test/', train=False, transform=transform, download=True)

        # Loading the test data and thus converting them into a test_loader
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)

    return test_loader, train_x, train_y


def initialize(model_id):
    # dataset = utils.get_parameter(param="dataset")
    model_name = utils.get_parameter(param="model")

    local_model =  models.load_model(model_name, model_id)

    local_model.save_checkpoint(os.getenv("TMP_FOLDER"))

    # torch.save(local_model.get_state_dict(), os.getenv("TMP_FOLDER") + modelID + ".pt")

    return local_model


def evaluate(local_model, test_loader, loss, attack=0):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    results_test = []
    # losses_test = []
    # acc_test = []

    num_peers = utils.get_parameter(param="num_peers")
    batch_size = utils.get_parameter(param="batch_size")
    dataset = utils.get_parameter(param="dataset")
    alpha = utils.get_parameter(param="alpha")
    iterations = utils.get_parameter(param="iterations")
    dc = utils.get_parameter(param="dc")
    attack_percentage = utils.get_parameter(param="attack_percentage")
    attack_type = utils.get_parameter(param="attack_type")

    # test_loader, _, _ = load_data()
    results, attack = test(local_model=local_model, test_loader=test_loader, attack=attack)
    results_test.append(results)
    # losses_test.append(test_loss)
    # acc_test.append(acc)

    results_str = utils.get_evaluation_results_str(results)

    message = 'average train loss %0.6g | test results %s' % (loss / num_peers, results_str)
    print(message)
    loop.run_until_complete(utils.send_log(message))
    asr = attack/(len(test_loader)*batch_size)
    message = 'attack success rate %0.3g' %(asr)
    print(message)
    loop.run_until_complete(utils.send_log(message))

    metric_filename = "{}_{}_{}_{}_{}_{}.csv".format(dataset, str(alpha), str(iterations), dc, attack_type, str(attack_percentage))


    log_message = os.getenv("MY_NAME") + "," + str(loss / num_peers) + "," + results_str + "!" + metric_filename
    loop.run_until_complete(utils.send_log(log_message))
    return results, asr

def get_all_local_data_loaders(shuffle=True):
    my_id = int(os.getenv("MY_ID"))
    dataset = utils.get_parameter("dataset")
    num_neg = int(utils.get_parameter("num_neg"))
    batch_size = int(utils.get_parameter("batch_size"))
    nentity = utils.get_parameter("metadata")["num_entities"]

    train_data_path = os.path.join(os.getenv("DATA_FOLDER"), dataset, str(my_id), "train.pkl")
    valid_data_path = os.path.join(os.getenv("DATA_FOLDER"), dataset, str(my_id), "valid.pkl")
    test_data_path = os.path.join(os.getenv("DATA_FOLDER"), dataset, str(my_id), "test.pkl")

    with open(train_data_path, "rb") as train_f, open(valid_data_path, "rb") as valid_f, open(test_data_path, "rb") as test_f:
        train_data = pickle.load(train_f)
        valid_data = pickle.load(valid_f)
        test_data = pickle.load(test_f)

    nrelation = len(np.unique(train_data['edge_type_ori']))

    train_triples = np.stack((train_data['edge_index_ori'][0],
                                train_data['edge_type_ori'],
                                train_data['edge_index_ori'][1])).T

    valid_triples = np.stack((valid_data['edge_index_ori'][0],
                                valid_data['edge_type_ori'],
                                valid_data['edge_index_ori'][1])).T

    test_triples = np.stack((test_data['edge_index_ori'][0],
                                test_data['edge_type_ori'],
                                test_data['edge_index_ori'][1])).T

    client_mask_ent = np.setdiff1d(np.arange(nentity),
                                    np.unique(train_data['edge_index_ori'].reshape(-1)), assume_unique=True)

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, num_neg)
    valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_ent)
    test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_ent)

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=TestDataset.collate_fn
    )

    ent_freq = torch.zeros(nentity)
    for e in train_data['edge_index_ori'].reshape(-1):
        ent_freq[e] += 1

    return train_dataloader, valid_dataloader, test_dataloader, ent_freq


def get_entity_freq():
    nentity = utils.get_parameter("metadata")["num_entities"]
    my_id = int(os.getenv("MY_ID"))
    dataset = utils.get_parameter("dataset")
    train_data_path = os.path.join(os.getenv("DATA_FOLDER"), dataset, str(my_id), "train.pkl")

    with open(train_data_path, "rb") as train_f:
        train_data = pickle.load(train_f)

    ent_freq = torch.zeros(nentity)
    for e in train_data['edge_index_ori'].reshape(-1):
        ent_freq[e] += 1

    return ent_freq

def train(local_model, train_loader, alpha="100", attack_type="lf"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if attack_type not in ["backdoor", "lf", "untargeted", "untargeted_sybil"]:
        message = "[x] ERROR: attack type not recognized :)"
        print(message)
        loop.run_until_complete(utils.send_log(message))
        return

    if alpha not in ["100","0.05","10"]:
        message = "[x] ERROR: alpha not recognized :)"
        print(message)
        loop.run_until_complete(utils.send_log(message))
        return

    dataset = utils.get_parameter(param="dataset")
    num_peers = utils.get_parameter(param="num_peers")
    epochs = utils.get_parameter(param="epochs")
    batch_size = utils.get_parameter(param="batch_size")
    model_name = utils.get_parameter(param="model")

    attack = 0
    loss = 0
    my_id = int(os.getenv("MY_ID"))
    # train_loader = get_train_data_loader(shuffle=True)

    # if dataset == "MNIST":
    #     train_obj = pickle.load(open(os.getenv("DATA_FOLDER") + dataset + "/" + str(my_id) + "/train_" + alpha +"_.pickle", "rb"))
    #     x = torch.stack(train_obj.x)
    #     y = torch.tensor(train_obj.y)
    #     dat = TensorDataset(x, y)
    #     train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)

    dishonest_peers = utils.get_dishonest_peers()
    if os.getenv("MY_ID") in dishonest_peers:
        attack += 1
        loss, local_model = peer_update(local_model=local_model, train_loader=train_loader, epoch=epochs, attack_type=attack_type)
    else:
        loss, local_model = peer_update(local_model=local_model, train_loader=train_loader, epoch=epochs)

    return loss, attack, local_model


def load_weights_into_model(weights):
    model_name = utils.get_parameter(param="model")

    local_model = models.load_model(model_name)
    local_model.set_state_dict(weights)

    # if dataset == "MNIST":
    #     local_model =  models.SFMNet(784, 10)
    #     state_dict = local_model.state_dict()
    #     state_dict['fc.weight'] = weights['fc.weight']
    #     state_dict['fc.bias'] = weights['fc.bias']
    #     local_model.load_state_dict(state_dict)

    return local_model


def get_optimizer(local_model):
    lr = utils.get_parameter(param="lr")

    # state_dict = local_model.get_state_dict()



    optimizer = optim.Adam([{'params': local_model.relation_embedding},
                            {'params': local_model.entity_embedding}], lr=lr)
    # dataset = utils.get_parameter(param="dataset")
    # opt = None
    # if dataset == "MNIST":
    #     lr = 0.01
    #     opt = optim.SGD(local_model.parameters(), lr=lr)
    # return opt

    return optimizer


def load_local_model():
    dataset = utils.get_parameter(param="dataset")
    model = None
    if dataset == "MNIST":
        model = SFMNet(784, 10)
    return model


def learn_locally(model_id):
    model_name = utils.get_parameter("model")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print(os.getenv("MY_NAME"), "Learning")
    # loop.run_until_complete(utils.send_log("Learning"))

    model_state_path = utils.get_model_state_path(model_id)
    print(model_state_path)
    if exists(model_state_path):
        message = "Weights to Train From = {}".format(str(len(model_state_path)))
        print(message)
        loop.run_until_complete(utils.send_log(message))

        print(os.getenv("MY_NAME"), "Training")

        # local_state_dict = torch.load(model_state_path)
        local_model = models.load_model(model_name, model_id)
        local_model.load_checkpoint(utils.get_model_state_dir_path())

        # load train, valid, test dataloaders
        train_loader, valid_dataloader, test_dataloader, entity_freq = get_all_local_data_loaders()

        # acc, asr = evaluate(local_model=local_model, test_loader=test_dataloader, loss=0, attack=0)

        loss, attack, local_model = train(
            local_model=local_model,
            train_loader=train_loader,
            alpha=utils.get_parameter(param="alpha"),
            attack_type=utils.get_parameter(param="attack_type"),
        )

        acc, asr = evaluate(local_model=local_model, test_loader=test_dataloader, loss=loss, attack=attack)
    else:
        print(os.getenv("MY_NAME"), "No weights to train from!")
        # loop.run_until_complete(utils.send_log("No weights to train from!"))


def get_publishing_data(local_model, local_entity_freq):
    data = OrderedDict()
    data["entity_embedding"] = local_model.entity_embedding
    data["relation_embedding"] = local_model.relation_embedding
    data["entity_freq"] = local_entity_freq

    return data


def learn(model_id):
    model_name = utils.get_parameter(param="model")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print(os.getenv("MY_NAME"), "Learning")
    loop.run_until_complete(utils.send_log("Learning"))

    state_dicts, indices, parents = utils.get_weights_to_train(model_id=model_id)

    # load local model
    model_state_path = utils.get_model_state_path(model_id)
    if exists(model_state_path) and len(state_dicts) > 0:
        local_state_dict = torch.load(model_state_path)
        state_dicts.append(local_state_dict)

    # aggregate local model with weights
    aggregated_state_dict = Aggregator()(state_dicts)
    local_model = models.load_model(model_name)
    local_model.set_state_dict(aggregated_state_dict)
    local_model.requires_grad_()

    # peers_models = []
    # for w in state_dicts:
    #     m = load_weights_into_model(w)
    #     peers_models.append(m)


    # if exists(model_state_path) and len(state_dicts) > 0:
    #     w = torch.load(model_state_path)
    #     m = load_weights_into_model(w)
    #     peers_models.append(m)

    # local_model = aggregate(peers_weights=peers_models, peers_indices=indices)

    train_loader, valid_dataloader, test_dataloader, entity_freq = get_all_local_data_loaders()

    message = "Weights to Train From = " + str(len(state_dicts))
    print(message)
    loop.run_until_complete(utils.send_log(message))
    if len(state_dicts) > 0:
        print(os.getenv("MY_NAME"), "Training")
        loss, attack, local_model = train(
            local_model=local_model,
            train_loader=train_loader,
            alpha=utils.get_parameter(param="alpha"),
            attack_type=utils.get_parameter(param="attack_type"),
        )
        # loss, attack, local_model = train(
        #     local_model=local_model,
        #     alpha=utils.get_parameter(param="alpha"),
        #     attack_type=utils.get_parameter(param="attack_type"),
        # )

        attacker = False
        dishonest_peers = utils.get_dishonest_peers()
        if os.getenv("MY_ID") in dishonest_peers:
            attacker = True

        acc = 0.0
        results = None
        if attacker == False:
            print(os.getenv("MY_NAME"), "Evaluating")
            loop.run_until_complete(utils.send_log("Evaluating"))
            results, asr = evaluate(local_model=local_model, test_loader=test_dataloader, loss=loss, attack=attack)
            # results, asr = evaluate(local_model=local_model, loss=loss, attack=attack)

        # if acc >= utils.get_my_latest_accuracy() or attacker:
        print(os.getenv("MY_NAME"), "Publishing")
        loop.run_until_complete(utils.send_log("Publishing"))
        local_model.save_checkpoint(utils.get_model_state_dir_path())
        # torch.save(local_model.state_dict(), os.getenv("TMP_FOLDER") + model_id + ".pt")
        blockID = utils.publish_model_update(
            modelID=model_id,
            weights=local_model.entity_embedding.detach().numpy(),
            accuracy=acc,
            parents=parents,
            model=get_publishing_data(local_model, entity_freq)
        )
        results_str = utils.get_evaluation_results_str(results)
        print(os.getenv("MY_NAME"), acc, blockID)
        utils.store_my_latest_accuracy(accuracy=results_str)
        loop.run_until_complete(utils.send_log(results_str + " " + str(blockID)))
    else:
        print(os.getenv("MY_NAME"), "No weights to train from!")
        loop.run_until_complete(utils.send_log("No weights to train from!"))
