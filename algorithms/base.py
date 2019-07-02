import os
import inspect
import random
import numpy 
import torch
import torch.optim as optim
from algorithms import HyperParameter
from utils.models import Policy, Value
from utils.memory import Memory
from utils.env import getEnvInfo

import matplotlib.pyplot as plt

def offPolicy(update):
    def wrapper(*args):
        alg = args[0]
        if len(alg.memory) > alg.param.BATCH_SIZE:
            transitions = alg.memory.sammple(alg.param.BATCH_SIZE)
            alg.offPolicyData = transitions
            update(alg)
    return wrapper
            
def onPolicy(update):
    def wrapper(*args):
        alg = args[0]
        loss , entropy = None, None
        if alg.steps % alg.param.BATCH_SIZE == 0:
            rollouts = alg.memory.replay()
            alg.onPolicyData = rollouts
            loss, entropy = update(alg)
            alg.memory.clear()
        return loss, entropy
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
        self.steps = 1

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
    

class ActorCritic(BaseRL):
    def __init__(self, env):
        super().__init__(env)
        self.actor = Policy(self.param.ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space)
        self.critic = Value(self.param.ARCHITECTURE, self.param.ACTIVATION)
        try:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.ACTOR_LEARNING_RATE)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.param.CRITIC_LEARNING_RATE)
        except AttributeError:
            print("ACTOR_LEARNING_RATE and CRITIC_LEARNING_RATE need to be specified in parameters.json")
    
    def act(self, state, exploit=False):
        ''' '''
        action = self.actor(torch.from_numpy(state).float(), exploit=exploit)
        next_state, reward, done, _ = self.env.step(action.numpy())
        if not exploit:
            self.memory.push(state, action, reward, next_state, done) 
            self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done
    
    def value(self, state):
        with torch.no_grad():
            value = self.critic(torch.from_numpy(state).float())
        return value.item()
    
    def optimize_critic(self, loss):
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
    
    def optimize_actor(self, loss):
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()


class ActorOnly(BaseRL):
    def __init__(self, env):
        super().__init__(env)
        self.actor = Policy(self.param.ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space)
        try:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.ACTOR_LEARNING_RATE)
        except AttributeError:
            print("ACTOR_LEARNING_RATE need to be specified in parameters.json")
    
    def act(self, state, exploit=False):
        ''' '''
        action = self.actor(torch.from_numpy(state).float(), exploit=exploit)
        next_state, reward, done, _ = self.env.step(action.numpy())
        self.memory.push(state, action, reward, next_state, done) 
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done
    
    def optimize_actor(self, loss):
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()


class CriticOnly(BaseRL):
    def __init__(self, env):
        super().__init__(env)

