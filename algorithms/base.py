import os
import random
import numpy 
import torch
from algorithms import HyperParameter
from utils.env import getEnvInfo

# TODO: 
# Naming: 
# BaseRL
# BasePG
# BaseQL

# tr_step
# clipped_step
# vanilla_step

class RLAlgorithm():
    def __init__(self, env):
        self.name = None
        self.env = env
        self.rng = random.Random()
        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.dataCollection = 'On-Policy'

    def act(self, state):
        pass
    
    def learn(self):
        pass
    
    def seed(self, seed):
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        self.rng = random.Random(seed)


    def reset(self):
        self.__init__(self.env)
    

class PolicyGradient(RLAlgorithm):
    def __init__(self, env):
        super().__init__(env)


class QLearning(RLAlgorithm):
    def __init__(self, env):
        super().__init__(env)