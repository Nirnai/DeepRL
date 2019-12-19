import time
import os
import inspect
import numpy as np
import torch
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from algorithms import HyperParameter, Buffer, ReplayBuffer

def getEnvInfo(env):
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    return state_dim, action_dim

class BaseRL(metaclass=ABCMeta):
    def __init__(self, env, param=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env = env
        self.device = torch.device(device)
        self.rng = np.random.RandomState(0)
        self.actor = None
        self.critic = None
        self.state_dim, self.action_dim = getEnvInfo(env)
        if param is None:
            self.param = self.load_parameters()
        else:
            self.param = param
        models = ['value', 'qvalue', 'policy']        
        for model in models: 
            if(hasattr(self.param, model)):
                attr = getattr(self.param, model)
                if not 'STATE_DIM' in attr.keys():
                    attr['STATE_DIM'] = self.state_dim
                    attr['ACTION_DIM'] = self.action_dim
                    if 'ARCHITECTURE' in attr.keys():
                        attr['ARCHITECTURE'].insert(0, self.state_dim)
                        attr['ARCHITECTURE'].append(self.action_dim)
        super(BaseRL, self).__init__(env, self.rng, self.param, device)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def load_parameters(self):
        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        return HyperParameter(parameters_file)

    def save_model(self, path):
        path += '/actor_model.pt'
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        path += '/actor_model.pt'
        self.actor.load_state_dict(torch.load(path))
        

    def seed(self, seed):
        pass
        # torch.manual_seed(seed)
        # self.rng.seed(seed)

    def reset(self):
        self.__init__(self.env, param=self.param)

class OnPolicy():
    def __init__(self, env, rng, param, device):
        self.memory = Buffer(param.BATCH_SIZE,param.GAMMA,param.LAMBDA, param.TAU, env, device)
        self.batch_size = param.BATCH_SIZE
    
    @classmethod
    def loop(cls, f):
        def wrap(self, *args):
            metrics = None
            if len(self.memory) == self.batch_size:
                rollouts = self.memory.replay()
                self.onPolicyData = rollouts
                metrics = f(self)
            return metrics
        return wrap


class OffPolicy():
    def __init__(self, env, rng, param, device):
        self.memory = ReplayBuffer(param.MEMORY_SIZE,rng,env,device)
        self.batch_size = param.BATCH_SIZE
        self.update_steps = param.UPDATE_STEPS

    @classmethod
    def loop(cls, f):
        def wrap(self, *args):
            metrics = None
            if len(self.memory) >= self.batch_size * self.update_steps and self.steps % self.update_steps == 0:
                self.offPolicyData = self.memory.sample(self.batch_size * self.update_steps)
                metrics = f(self)
            return metrics
        return wrap


