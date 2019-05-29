import os
import inspect
import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from copy import deepcopy
from utils.models import Policy, Value
from utils.memory import RolloutBuffer
from utils.env import getEnvInfo
from algorithms import RLAlgorithm, HyperParameter

class PPO(RLAlgorithm):
    def __init__(self, env):
        super(PPO, self).__init__(env)
        self.name = "PPO"
        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        self.param = HyperParameter(parameters_file)
        self.param.ARCHITECTURE[ 0 ] = self.state_dim
        self.param.ARCHITECTURE[-1 ] = self.action_dim
        architecture = self.param.ARCHITECTURE
        activation = self.param.ACTIVATION

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.actor = Policy(architecture, activation, action_space=self.action_space)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
        self.critic = Value(architecture, activation)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.param.LEARNING_RATE)
        self.rolloutBuffer = RolloutBuffer()

        self.steps = 0
        self.done = False
    
    def act(self, state, exploit=False):
        ''' '''
        policy = self.actor(torch.from_numpy(state).float())
        if exploit:
            action = policy.mean.detach()
        else:
            action = policy.sample()
        next_state, reward, done, _ = self.env.step(action.numpy()) 
        self.rolloutBuffer.push(state, action, reward, next_state, done)
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done


    def learn(self):
        if self.steps % self.param.BATCH_SIZE == 0:
            batch = self.rolloutBuffer.sample()
            states = torch.Tensor(batch.state)
            actions = torch.stack(batch.action)
            rewards = torch.Tensor(batch.reward)
            next_states = torch.Tensor(batch.next_state)
            mask = torch.Tensor(batch.mask)

            for _ in range(4):
                advantages = self.gae(states, next_states[-1], rewards, mask)
                advantages = (advantages - advantages.mean()) / advantages.std()
                # Optimizer Critic
                self.critic_optim.zero_grad()
                value_loss = advantages.pow(2.).mean()
                value_loss.backward()
                self.critic_optim.step()

                with torch.no_grad():
                    actor_old = deepcopy(self.actor)
                    old_policy = actor_old(states)
                    old_log_probs = old_policy.log_prob(actions).squeeze()       

                curr_policy = self.actor(states)
                curr_log_probs = curr_policy.log_prob(actions).squeeze()

                # Optimize Actor
                self.actor_optim.zero_grad()
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.param.CLIP,  1 + self.param.CLIP) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss.backward()
                self.actor_optim.step()
            del self.rolloutBuffer.memory[:]


    def gae(self, states, next_state ,rewards, mask):
        '''  Generaized Advantage Estimation '''
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_value = self.critic(next_state)
        values = torch.cat((values, next_value))
        advantages = [0] * (len(rewards)+1)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.param.GAMMA * values[t+1] * mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * mask[t] * advantages[t+1]
        return torch.stack(advantages[:-1])


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env)
