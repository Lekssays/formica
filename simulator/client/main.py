import argparse
import json
import learning
import utils
import os
import pickle

from os.path import exists


class client:
    def __init__(self, client_id, x, y):
        self.client_id = client_id
        self.x= x
        self.y=y

    def write_out(self, dataset, alpha, mode='train'):
        if not os.path.isdir(dataset):
            os.mkdir(dataset)
        if not os.path.isdir(dataset+'/'+str(self.client_id)):
            os.mkdir(dataset+'/'+str(self.client_id))

        with open(dataset+'/'+str(self.client_id)+'/'+mode+'_'+str(alpha)+'_'+'.pickle', 'wb') as f:
            pickle.dump(self, f)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset',
                        dest = "dataset",
                        help = "Dataset: MNIST, CIFAR, KDD",
                        default = "MNIST",
                        required = True)
    parser.add_argument('-m', '--model',
                        dest = "model",
                        help = "Model: TransE",
                        default = "TransE",
                        required = True)
    parser.add_argument('-am', '--agg_mode',
                        dest = "agg_mode",
                        help = "Modes: Isolation",
                        default = "TransE",
                        required = True)
    parser.add_argument('-p', '--peers',
                        dest = "num_peers",
                        help = "Number of Peers ",
                        default = 100,
                        required = False)
    parser.add_argument('-i', '--iterations',
                        dest = "iterations",
                        help = "Number of Iterations",
                        default = 100,
                        required = False)
    parser.add_argument('-e', '--epochs',
                        dest = "epochs",
                        help = "Number of Training Epochs",
                        default = 1,
                        required = False)
    parser.add_argument('-bz', '--batch_size',
                        dest = "batch_size",
                        help = "Batch Size",
                        default = 50,
                        required = False)
    parser.add_argument('-tr', '--threshold',
                        dest = "threshold",
                        help = "Threshold",
                        default = 0.1,
                        required = False)
    parser.add_argument('-al', '--alpha',
                        dest = "alpha",
                        help = "Dirichlet distribution factor",
                        default = 100,
                        required = False)
    parser.add_argument('-at', '--attack_type',
                        dest = "attack_type",
                        help = "Attack type: lf , backdoor, untargeted",
                        default = "lf",
                        required = False)
    parser.add_argument('-ap', '--attack_percentage',
                        dest = "attack_percentage",
                        help = "Attack percentage",
                        default = "0",
                        required = False)
    parser.add_argument('-dc', '--dynamic_committee',
                        dest = "dc",
                        help = "Enable dynamic committee",
                        default = "true",
                        required = False)
    parser.add_argument('-adt', '--adversarial_temperature',
                        dest = "adversarial_temperature",
                        default=1.0,
                        type=float)
    parser.add_argument('-n', '--num_neg',
                        dest = "num_neg",
                        help = "number of negative sample for training KGE",
                        default=256,
                        type=int)

    parser.add_argument('--lr', "--learning_rate",
                        dest="learning_rate",
                        help='learning rate for training KGE on FedE, Isolation or Collection',
                        default=0.001,
                        type=int)


    # args = parser.parse_args()

    # print(args)
    return parser.parse_args()


def generate_config():
    dataset = parse_args().dataset
    num_peers = int(parse_args().num_peers)
    epochs = int(parse_args().epochs)
    batch_size = int(parse_args().batch_size)
    alpha = str(parse_args().alpha)
    attack_type = str(parse_args().attack_type)
    iterations = int(parse_args().iterations)
    dc = parse_args().dc
    attack_percentage = int(parse_args().attack_percentage)
    model_name = parse_args().model
    aggregation_mode = parse_args().agg_mode
    metadata = utils.get_dataset_metadata(dataset)
    adversarial_temperature = parse_args().adversarial_temperature
    num_neg = parse_args().num_neg
    lr = parse_args().learning_rate

    config = {
        'dataset': dataset,
        'model': model_name,
        'aggregation_mode': aggregation_mode,
        'num_peers': num_peers,
        'iterations': iterations,
        'epochs': epochs,
        'batch_size': batch_size,
        'alpha': alpha,
        'attack_type': attack_type,
        'dc': dc,
        'attack_percentage': attack_percentage,
        'metadata': metadata,
        'adversarial_temperature': adversarial_temperature,
        'num_neg': num_neg,
        'lr': lr,
    }

    f = open('config.json', 'w')
    f.write(json.dumps(config))
    f.close()

    return dataset


def main():
    generate_config()

    model_id = "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4"
    agg_mode = utils.get_parameter("aggregation_mode")

    if not exists(os.getenv("TMP_FOLDER") + model_id + ".dat"):
        local_model = learning.initialize(model_id)
        entity_freq = learning.get_entity_freq()
        block_id = utils.publish_model_update(
            modelID=model_id,
            parents=[],
            weights=local_model.entity_embedding.detach().numpy(),
            model=learning.get_publishing_data(local_model, entity_freq),
            accuracy=0.0,
        )
        utils.store_my_latest_accuracy(accuracy=0.00)
        utils.store_weight_id(modelID=model_id, blockID=block_id)

    learning.learn(model_id=model_id)


if __name__ == "__main__":
    main()
