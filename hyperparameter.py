import json
import yaml
import torch

class HyperParameter:
    def __init__(self, path=None):
        # # Network Architecture for Neural Network (e.g. [4,128,2]) 
        # self.NETWORK_ARCHITECTURE = None

        # # Activations Function
        # # Possible activations in Pytorch:
        # # 'CELU', 'ELU', 'F', 'GLU', 'Hardshrink', 'Hardtanh', 'LeakyReLU', 
        # # 'LogSigmoid', 'LogSoftmax', 'PReLU', 'RReLU', 'ReLU', 'ReLU6', 'SELU', 
        # # 'Sigmoid', 'Softmax', 'Softmax2d', 'Softmin', 'Softplus', 'Softshrink', 
        # # 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold'
        # self.ACTIVATION = None

        # # Loss Function
        # # Possible Loss functions in Pytorch
        # # 'BCELoss', 'BCEWithLogitsLoss', 'CTCLoss', 'CosineEmbeddingLoss', 'CrossEntropyLoss', 
        # # 'HingeEmbeddingLoss', 'KLDivLoss', 'L1Loss', 'LogSoftmax', 'MSELoss', 'MarginRankingLoss', 
        # # 'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'NLLLoss', 'NLLLoss2d', 
        # # 'PoissonNLLLoss', 'Sequential', 'SmoothL1Loss', 'SoftMarginLoss', 'TripletMarginLoss'
        # self.LOSS = None
        if path is not None:
            self.load_parameters(path)

    def __repr__(self):
        return "<{klass} \n {attrs}>".format(
            klass=self.__class__.__name__,
            attrs=" ".join("{}={!r} \n".format(k, v) for k, v in self.__dict__.items()),
            )

    def save_parameters(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4, sort_keys=True)
    
    def load_parameters(self, path):
        with open(path) as infile:
            parameters = json.load(infile)
            for key in parameters:
                self.__setattr__(key, parameters[key])


    # TODO: Epsilon Decay should be part of evaluation


    # TODO: Seeds identifies: 1.torch, 2.numpy, 3.random --> set as parameters 