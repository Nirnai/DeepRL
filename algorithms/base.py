import os
import inspect
import random
import numpy 
import torch
import torch.optim as optim
from algorithms import HyperParameter
from utils.models import Policy
from utils.memory import Memory
from utils.env import getEnvInfo


# TODO: 
# Naming: 
# BaseRL
# BasePG
# BaseQL

# tr_step
# clipped_step
# vanilla_step

def onPolicy(update):
    def wrapper(*args):
        alg = args[0]
        if alg.steps % alg.param.BATCH_SIZE == 0:
            rollouts = alg.memory.replay()
            alg.onPolicyData = rollouts
            update(alg)
            alg.memory.clear()
    return wrapper  

class BaseRL():
    def __init__(self, env):
        self.name = None
        self.env = env
        self.rng = random.Random()
        self.param = self.laod_parameter_file()
        
        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        if(hasattr(self.param, 'ARCHITECTURE')):
            self.param.ARCHITECTURE[ 0] = self.state_dim
            self.param.ARCHITECTURE[-1] = self.action_dim

        if(hasattr(self.param, 'MEMORY_SIZE')):
            self.memory = Memory(self.param.MEMORY_SIZE, self.rng)
        elif(hasattr(self.param, 'BATCH_SIZE')):
            self.memory = Memory(self.param.BATCH_SIZE, self.rng)
        else:
            raise Exception('Either MEMORY_SIZE or BATCH_SIZE needs to be specified in parameters.json')

        self.steps = 0

    def act(self, state, exploit=False):
        pass
    
    def learn(self):
        pass
            

    def laod_parameter_file(self):
        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        return HyperParameter(parameters_file)

    def seed(self, seed):
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        self.rng = random.Random(seed)

    def reset(self):
        self.__init__(self.env)
    

class BasePG(BaseRL):
    def __init__(self, env):
        super().__init__(env)
        self.policy = Policy(self.param.ARCHITECTURE, 
                            self.param.ACTIVATION, 
                            action_space=self.action_space)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.param.LEARNING_RATE)

    
    def act(self, state, exploit=False):
        ''' '''
        action = self.policy(torch.from_numpy(state).float(), exploit=exploit)
        next_state, reward, done, _ = self.env.step(action.numpy())
        self.memory.push(state, action, reward, next_state, done) 
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done



class BaseQL(BaseRL):
    def __init__(self, env):
        super().__init__(env)

