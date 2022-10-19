import asyncio
import pickle
import os
import models
import utils
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from os.path import exists
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from models import SFMNet

torch.backends.cudnn.benchmark=True
torch.manual_seed(42)
np.random.seed(42)


def peer_update(local_model, train_loader, epoch=5, attack_type=None):
    print(os.getenv("MY_NAME"), "attack_type", attack_type)
    dataset = utils.get_parameter(param="dataset")
    optimizer = get_optimizer(local_model)
    local_model.train()
    for _ in range(epoch):
        for _, (data, target) in enumerate(train_loader):
            data, target = data, target
            optimizer.zero_grad()
            output = local_model(data)

            # Attack can be added here depending on the dataset
            if attack_type is not None:
                if dataset == "MNIST":
                    for i, t in enumerate(target):
                        if attack_type == 'lf':  # label flipping
                            if t == 1:
                                target[i] = torch.tensor(7)
                        elif attack_type == 'backdoor':
                            target[i] = 1  # set the label
                            data[:, :, 27, 27] = torch.max(data)  # set the bottom right pixel to white.
                        elif attack_type == 'untargeted':
                            target[i] = random.randint(0, 9)
                        elif attack_type == "untargeted_sybil":  # untargeted with sybils
                            target[i] = 0
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    return loss.item(), local_model


def aggregate(peers_indices, peers_weights=[]):
    if len(peers_weights) == 0:
        return
    phi = utils.get_phi()
    if phi is None:
        return
    global_dict = peers_weights[-1].state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([peers_weights[idx].state_dict()[k].float()* phi[peers_indices[idx]] for idx in range(0, len(peers_indices))], 0).mean(0)
    peers_weights[-1].load_state_dict(global_dict)
    return peers_weights[-1]


def test(local_model, test_loader, attack):
    """This function test the global model on test data and returns test loss and test accuracy """
    local_model.eval()
    test_loss = 0
    correct = 0
    dataset = utils.get_parameter(param="dataset")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = local_model(data)
            
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

            if dataset == "MNIST":
                for i, t in enumerate(target):
                    if t == 1 and pred[i] == 7:
                        attack += 1

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc, attack


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


def initialize(modelID):
    dataset = utils.get_parameter(param="dataset")

    if dataset == "MNIST":
        local_model =  models.SFMNet(784, 10)

    torch.save(local_model.state_dict(), os.getenv("TMP_FOLDER") + modelID + ".pt")

    return local_model


def evaluate(local_model, loss, attack=0):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    losses_test = []
    acc_test = []

    num_peers = utils.get_parameter(param="num_peers")
    batch_size = utils.get_parameter(param="batch_size")
    dataset = utils.get_parameter(param="dataset")
    alpha = utils.get_parameter(param="alpha")
    iterations = utils.get_parameter(param="iterations")
    dc = utils.get_parameter(param="dc")
    attack_percentage = utils.get_parameter(param="attack_percentage")
    attack_type = utils.get_parameter(param="attack_type")

    test_loader, _, _ = load_data()
    test_loss, acc, attack = test(local_model=local_model, test_loader=test_loader, attack=attack)
    losses_test.append(test_loss)
    acc_test.append(acc)

    message = 'average train loss %0.6g | test loss %0.6g | test acc: %0.6f' % (loss / num_peers, test_loss, acc)
    print(message)
    loop.run_until_complete(utils.send_log(message))
    asr = attack/(len(test_loader)*batch_size)
    message = 'attack success rate %0.3g' %(asr)
    print(message)
    loop.run_until_complete(utils.send_log(message))
    
    metric_filename = "{}_{}_{}_{}_{}_{}.csv".format(dataset, str(alpha), str(iterations), dc, attack_type, str(attack_percentage))
    log_message = os.getenv("MY_NAME") + "," + str(loss / num_peers) + "," + str(test_loss) + "," + str(acc) + "," + str(asr) + "!" + metric_filename
    loop.run_until_complete(utils.send_log(log_message))
    return acc, asr


def train(local_model, alpha="100", attack_type="lf"):
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
    
    attack = 0
    loss = 0
    my_id = int(os.getenv("MY_ID"))
    if dataset == "MNIST":
        train_obj = pickle.load(open(os.getenv("DATA_FOLDER") + dataset + "/" + str(my_id) + "/train_" + alpha +"_.pickle", "rb"))
        x = torch.stack(train_obj.x)
        y = torch.tensor(train_obj.y)
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)

    dishonest_peers = utils.get_dishonest_peers()
    if os.getenv("MY_ID") in dishonest_peers:
        attack += 1
        loss, local_model = peer_update(local_model=local_model, train_loader=train_loader, epoch=epochs, attack_type=attack_type)
    else:
        loss, local_model = peer_update(local_model=local_model, train_loader=train_loader, epoch=epochs)

    return loss, attack, local_model


def load_weights_into_model(weights):
    dataset = utils.get_parameter(param="dataset")
    if dataset == "MNIST":
        local_model =  models.SFMNet(784, 10)
        state_dict = local_model.state_dict()
        state_dict['fc.weight'] = weights['fc.weight']
        state_dict['fc.bias'] = weights['fc.bias']
        local_model.load_state_dict(state_dict)

    return local_model


def get_optimizer(local_model):
    dataset = utils.get_parameter(param="dataset")
    opt = None
    if dataset == "MNIST":
        lr = 0.01
        opt = optim.SGD(local_model.parameters(), lr=lr)
    return opt


def load_local_model():
    dataset = utils.get_parameter(param="dataset")
    model = None
    if dataset == "MNIST":
        model = SFMNet(784, 10)
    return model


def learn(modelID):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print(os.getenv("MY_NAME"), "Learning")
    loop.run_until_complete(utils.send_log("Learning"))
    
    weights_ids, indices, parents = utils.get_weights_to_train(modelID=modelID)

    peers_weights = []
    for wid in weights_ids:
        m = load_weights_into_model(wid)
        peers_weights.append(m)

    if exists(os.getenv("TMP_FOLDER") + modelID + ".pt") and len(weights_ids) > 0:
        w = torch.load(os.getenv("TMP_FOLDER") + modelID + ".pt")
        m = load_weights_into_model(w)
        peers_weights.append(m)

    local_model = aggregate(peers_weights=peers_weights, peers_indices=indices)

    message = "Weights to Train From = " + str(len(weights_ids))
    print(message)
    loop.run_until_complete(utils.send_log(message))
    if len(weights_ids) > 0:
        print(os.getenv("MY_NAME"), "Training")
        loss, attack, local_model = train(
            local_model=local_model,
            alpha=utils.get_parameter(param="alpha"),
            attack_type=utils.get_parameter(param="attack_type"),
        )

        attacker = False
        dishonest_peers = utils.get_dishonest_peers()
        if os.getenv("MY_ID") in dishonest_peers:
            attacker = True

        acc = 0.0
        if attacker == False:
            print(os.getenv("MY_NAME"), "Evaluating")
            loop.run_until_complete(utils.send_log("Evaluating"))
            acc, asr = evaluate(local_model=local_model, loss=loss, attack=attack)    
        
        if acc >= utils.get_my_latest_accuracy() or attacker:
            print(os.getenv("MY_NAME"), "Publishing")
            loop.run_until_complete(utils.send_log("Publishing"))
            torch.save(local_model.state_dict(), os.getenv("TMP_FOLDER") + modelID + ".pt")
            blockID = utils.publish_model_update(
                modelID=modelID,
                weights=local_model.state_dict()['fc.weight'].cpu().numpy(),
                accuracy=acc,
                parents=parents,
                model=local_model.state_dict()
            )
            print(os.getenv("MY_NAME"), acc, blockID)
            utils.store_my_latest_accuracy(accuracy=acc)
            loop.run_until_complete(utils.send_log(str(acc) + " " + str(blockID)))
    else:
        print(os.getenv("MY_NAME"), "No weights to train from!")
        loop.run_until_complete(utils.send_log("No weights to train from!"))
