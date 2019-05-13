import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from algorithms.utils import Policy, Value
from collections import namedtuple

Transition = namedtuple('Transition', ('states',  'log_probs', 'rewards', 'next_states', 'dones'))


class A2C():
    def __init__(self, env, param):
        self.name = "A2C"
        self.env = env
        self.param = param
        self.rng = random.Random()

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.actor = Policy(self.param.ACTOR_ARCHITECTURE, self.param.ACTIVATION)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)

        self.critic = Value(self.param.CRITIC_ARCHITECTURE, self.param.ACTIVATION)
        self.critic_optimizer = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)

        self.rollout = []

        self.done = False
    
    def act(self, state):
        probs = self.actor(torch.from_numpy(state).float())
        m = dist.Categorical(probs)
        action = m.sample()
        next_state, reward, self.done, _ = self.env.step(action.numpy()) 

        self.rollout.append(Transition(state, 
                                       m.log_prob(action), 
                                       reward, 
                                       next_state, 
                                       self.done))

        return next_state, reward, self.done


    def learn(self):
        if self.done:
            rollout = Transition(*zip(*self.rollout))

            # Monte Carlo Targets
            returns = self.monte_carlo_target(rollout.rewards).squeeze().detach()

            # Bootstrapped Targets
            # returns = self.bootstrapped_target(rollout.rewards, rollout.next_states).squeeze().detach()

            values = self.critic(torch.Tensor(rollout.states)).squeeze()   
            advantages = returns - values
            log_probs = torch.stack(rollout.log_probs)

            value_loss = advantages.pow(2).mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            policy_loss = (- log_probs * advantages.detach()).sum()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            del self.rollout[:]

    
    def bootstrapped_target(self, rewards, next_states):
        rewards = torch.Tensor(rewards).squeeze()
        with torch.no_grad():
            next_values = self.critic(torch.Tensor(next_states)).squeeze()
            values = rewards + self.param.GAMMA * next_values
        return values
        

    def monte_carlo_target(self, rewards):
        R = 0
        V = []
        for r in rewards[::-1]:
            R = r + self.param.GAMMA * R
            V.insert(0, R)   
        V = torch.Tensor(V)
        # V = V - V.mean()
        return V


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)
