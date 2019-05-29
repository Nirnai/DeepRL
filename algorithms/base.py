import os
import random 
from algorithms import HyperParameter
from utils.env import getEnvInfo


class RLAlgorithm():
    def __init__(self, env):
        self.name = None
        self.env = env
        self.rng = random.Random()
        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)

    def act(self, state):
        pass
    
    def learn(self):
        pass