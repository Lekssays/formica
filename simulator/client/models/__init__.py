from .sfmnet import SFMNet
from .kge import KGEModel

def load_model(model_name):
    if model_name == "MNIST":
        return SFMNet(784, 10)
    elif model_name == "TransE":
        # default params from FedE
        args = {
            "gamma": 10.0,
            "epsilon": 2.0,
            "hidden_dim": 128,
        }
        return KGEModel(args, "TransE")

    raise Exception("{} is unsupported".format(model_name))