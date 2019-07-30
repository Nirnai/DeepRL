import time
import os
import inspect

# import random
import numpy as np

import torch
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from functools import wraps
from algorithms import HyperParameter
from utils.env import getEnvInfo
from utils.values_functions import Value, QValue
from utils.memory import Memory

class BaseRL(metaclass=ABCMeta):
    def __init__(self, env, **kw):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        # self._rng = random.Random()
        self._rng = np.random.RandomState(0)
        self._state_dim, self._action_dim, self._action_space = getEnvInfo(env)
        self.param = self.load_parameters()
        if(hasattr(self.param, 'ARCHITECTURE')):
            self.param.ARCHITECTURE[ 0] = self._state_dim
            self.param.ARCHITECTURE[-1] = self._action_dim
        super(BaseRL, self).__init__(self.param, self._rng, self.env)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def load_parameters(self):
        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        return HyperParameter(parameters_file)

    def seed(self, seed):
        torch.manual_seed(seed)
        self._rng.seed(seed)

    def reset(self):
        self.__init__(self.env)

class OnPolicy():
    def __init__(self, param, *args, **kw):
        self._memory = Memory(param.BATCH_SIZE, None)
        self._batch_size = param.BATCH_SIZE
    
    @classmethod
    def loop(cls, f):
        def wrap(self, *args):
            metrics = None
            if len(self._memory) == self._batch_size:
                rollouts = self._memory.replay()
                self.onPolicyData = rollouts
                metrics = f(self)
                self._memory.clear()
            return metrics
        return wrap


class OffPolicy():
    def __init__(self, param, rng, env, **kw):
        self._rng = rng
        self._memory = Memory(param.MEMORY_SIZE, rng, env)
        self._batch_size = param.BATCH_SIZE
        self._update_steps = param.UPDATE_STEPS
        super(OffPolicy, self).__init__(**kw)

    @classmethod
    def loop(cls, f):
        def wrap(self, *args):
            metrics = None
            if len(self._memory) >= self._batch_size * self._update_steps and self.steps % self._update_steps == 0:
                self.offPolicyData = self._memory.sample(self._batch_size * self._update_steps)
                metrics = f(self)
            return metrics
        return wrap


class QModel():
    def __init__(self, param):
        self.Q1 = QValue(param.ARCHITECTURE, param.ACTIVATION)
        self.Q2 = QValue(param.ARCHITECTURE, param.ACTIVATION)
        self.Q1_target = QValue(param.ARCHITECTURE, param.ACTIVATION)
        self.Q2_target = QValue(param.ARCHITECTURE, param.ACTIVATION)
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q1.state_dict())
        self.Q1_target.eval()
        self.Q2_target.eval()
        self.Q_optim = optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=param.LEARNING_RATE)
        self._tau = param.TAU
    
    def __call__(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)
    
    def target(self, state, action):
        return self.Q1_target(state, action), self.Q2_target(state, action)

    def optimize(self, loss):
        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()
        self.target_update()
    
    def target_update(self):
        for target_param, local_param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)
        for target_param, local_param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)

class VModel():   
    def __init__(self, param):
        self.V = Value(param.ARCHITECTURE, param.ACTIVATION)
        self.V_optim = optim.Adam(self.V.parameters(), lr=param.CRITIC_LEARNING_RATE)
    
    def __call__(self, state):
        return self.V(state)

    def optimize(self, loss):
        self.V_optim.zero_grad()
        loss.backward()
        self.V_optim.step()

