from .network import Classifier
from .baselines import SCRNet


def get_model(name, n_class):
    return {            
            'net1': Classifier(n_class=n_class)
           }[name]

