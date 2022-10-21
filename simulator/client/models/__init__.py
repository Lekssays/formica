from .sfmnet import SFMNet
from .kge import KGEModel
import utils

def load_model(model_id, model_name):
    config = utils.get_config()

    if model_name == "MNIST":
        return SFMNet(784, 10)
    elif model_name == "TransE":
        # default params from FedE

        gamma = config.get("gamma", 10.0)
        epsilon = config.get("epsilon", 2.0)
        hidden_dim = config.get("hidden_dim", 128)
        device_name = config.get("device", "cpu")

        # required params
        num_entities = config["metadata"]["n_entities"]
        num_relations = config["metadata"]["n_relations"]

        args = {
            "gamma": gamma,
            "epsilon": epsilon,
            "hidden_dim": hidden_dim,
            "device": device_name,
            "num_entities": num_entities,
            "num_relations": num_relations
        }
        return KGEModel(model_id, "TransE", args)

    raise Exception("{} is unsupported".format(model_name))