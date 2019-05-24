import random
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
# from algorithms.utils import Policy, ActionValue
from utils.helper import hard_target_update, soft_target_update
from utils.memory import ReplayBuffer
from utils.env import getEnvInfo
from utils.noise import OrnsteinUhlbeckNoise

class DDPG():
    def __init__(self, env, param):
        self.name = "DDPG"
        self.env = env
        self.param = param
        self.rng = random.Random()

        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.param.ACTOR_ARCHITECTURE[0] = self.state_dim
        self.param.CRITIC_ARCHITECTURE[0] = self.state_dim
        self.param.ACTOR_ARCHITECTURE[-1] = self.action_dim
        self.param.CRITIC_ARCHITECTURE[-1] = 1

        if self.param.SEED != None:
            self.seed(self.param.SEED)


        # self.actor = Policy(self.param.ACTOR_ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space, deterministic=True)
        # self.actor_target = Policy(self.param.ACTOR_ARCHITECTURE, self.param.ACTIVATION, action_space=self.action_space, deterministic=True)
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
        # hard_target_update(self.actor, self.actor_target)

        # self.critic = ActionValue(self.param.CRITIC_ARCHITECTURE, self.param.ACTIVATION)
        # self.critic_target = ActionValue(self.param.CRITIC_ARCHITECTURE, self.param.ACTIVATION)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.param.LEARNING_RATE)
        # hard_target_update(self.critic, self.critic)

        self.replay = ReplayBuffer(self.param.MEMORY_SIZE, self.rng)
        self.exploration_process = OrnsteinUhlbeckNoise(self.action_dim, self.rng)


    def act(self, state):
        pass
        # s = torch.from_numpy(state).float()
        # with torch.no_grad():
        #     action = self.actor(s)
        # exploration = torch.Tensor(self.exploration_process.sample())
        # action = (action + exploration).clamp(-1,1)

        # test = self.critic(s, action)

        # next_state, reward, done, _ = self.env.step(action.numpy()) 
        # self.replay.push(state, action, reward, next_state, done)
        # return next_state, reward, done


    def learn(self):
        pass
    
    def seed(self, seed):
        self.param.SEED = seed
        self.rng.seed(self.param.SEED)
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        
    
    def reset(self):
        self.__init__(self.env, self.param)


