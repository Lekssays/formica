from .sfmnet import SFMNet
from .kge import KGEModel
import utils

def load_model(model_name, model_id = ""):
    config = utils.get_config()

    if model_name == "TransE":
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
        model = KGEModel(model_id, "TransE", args)

    else:
        raise Exception("{} is unsupported".format(model_name))

    return model