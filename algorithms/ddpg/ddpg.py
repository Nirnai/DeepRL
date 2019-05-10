import random
import numpy
import torch
import torch.nn as nn

class DDPG():
    def __init__(self, env, param):
        self.name = "DDPG"
        self.env = env
        self.param = param
        self.rng = random.Random()
    
    def act(self, state):
        pass

    def learn(self):
        pass
    
    def seed(self, seed):
        self.param.SEED = seed
        self.rng.seed(self.param.SEED)
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        
    
    def reset(self):
        self.__init__(self.env, self.param)

    

class ActionValue(nn.Module):
    def __init__(self):
        super(ActionValue, self).__init__()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()