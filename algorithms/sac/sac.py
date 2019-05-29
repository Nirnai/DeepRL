import os
import inspect
import torch
import torch.optim as optim
from copy import deepcopy
from algorithms import RLAlgorithm, HyperParameter
from utils.models import Policy, Value, QValue
from utils.helper import soft_target_update
from utils.memory import ReplayBuffer

class SAC(RLAlgorithm):
    def __init__(self, env):
        super(SAC,self).__init__(env)
        self.name = "SAC"

        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        self.param = HyperParameter(parameters_file)
        self.param.ARCHITECTURE[ 0 ] = self.state_dim
        self.param.ARCHITECTURE[-1 ] = self.action_dim
        architecture = self.param.ARCHITECTURE
        activation = self.param.ACTIVATION

        self.actor = Policy(architecture, activation, action_space=self.action_space)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
        self.vcritic = Value(architecture, activation)
        self.vcritic_target = deepcopy(self.vcritic)
        self.vcritic_optim = optim.Adam(self.vcritic.parameters(), lr=self.param.LEARNING_RATE)
        self.qcritic1 = QValue(architecture, activation) 
        self.qcritic2 = QValue(architecture, activation) 
        self.qritics_optim = optim.Adam(list(self.qcritic1.parameters()) + list(self.qcritic2.parameters()), lr=self.param.LEARNING_RATE)
        
        self.replay_buffer = ReplayBuffer(self.param.MEMORY_SIZE, self.rng)

    
    def act(self, state):
        pass
    
    def learn(self):
        pass
    
    def seed(self, seed):
        pass

    def reset(self):
        pass