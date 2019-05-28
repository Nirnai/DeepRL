import inspect
import os

import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

from copy import deepcopy
from torch.distributions import kl_divergence
from algorithms import RLAlgorithm, HyperParameter
from utils.models import Policy, Value
from utils.memory import RolloutBuffer
from utils.env import getEnvInfo


class TRPO(RLAlgorithm):
    def __init__(self, env):
        super(TRPO, self).__init__(env)
        self.name = "TRPO"

        parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
        self.param = HyperParameter(parameters_file)
        self.param.ARCHITECTURE[ 0 ] = self.state_dim
        self.param.ARCHITECTURE[-1 ] = self.action_dim
        architecture = self.param.ARCHITECTURE
        activation = self.param.ACTIVATION

        self.actor = Policy(architecture, activation, action_space=self.action_space)
        self.actor_old = deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
        self.critic = Value(architecture, activation)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.param.LEARNING_RATE)
        self.rolloutBuffer = RolloutBuffer()


        self.steps = 0
        self.done = False

        
    def act(self, state, exploit=False):
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

            advantages = self.gae(states, next_states[-1], rewards, mask)
            
            # Optimize Critic            
            self.critic_optim.zero_grad()
            value_loss = advantages.pow(2.).mean()
            value_loss.backward()
            self.critic_optim.step()

            ####
            advantages = (advantages - advantages.mean()) / advantages.std()
            advantages = advantages.detach()
            
            p = self.actor(states)
            with torch.no_grad():
                p_old = self.actor_old(states)
            d_kl = kl_divergence(p, p_old).mean()

            # Compute Hessian Vector Product
            def Hx(x):
                # TODO: Possibility of damping
                grads = torch.autograd.grad(d_kl, self.actor.parameters(), create_graph=True),
                grads = torch.cat([grad.view(-1) for grad in grads])
                Jx = (grads * x).sum()
                Hx = torch.autograd.grad(Jx, self.actor.parameters())
                Hx = torch.cat([grad.view(-1) for grad in Hx])
                return Hx 


            log_probs = p.log_prob(actions).squeeze()
            log_probs_old = p_old.log_prob(actions).squeeze() 

            ratio = torch.exp(log_probs - log_probs_old)
            loss = (- ratio * advantages).mean()

            grads = torch.autograd.grad(loss, self.actor.parameters())

            gradient = self.actor.get_grads(loss)
            stepdir = self.conjugate_gradient(Hx, gradient)
            
            delta = self.param.DELTA
            natural_gradient = (2 * delta) / torch.dot(stepdir,self.Hx(states, stepdir))
            natural_gradient = torch.sqrt(natural_gradient) * stepdir

            
            # params = self.linesearch()

            for param in self.actor.parameters():
                param.data -= natural_gradient
           
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


    def conjugate_gradient(self, A, b):
        x = torch.zeros(b.size())
        r = b.clone()
        p = r.clone()
        rs = torch.dot(r,r)
        for i in range(b.shape[0]):
            if callable(A):
                Ap = A(p)
            else:
                Ap = torch.matmul(A,p)
            alpha = rs/(torch.dot(p, Ap))
            x += alpha * p
            r -= alpha * Ap
            rs_next = torch.dot(r, r)
            betta = rs_next / rs
            p = r + betta * p
            rs = rs_next
            if rs < 1e-10:
                break
        return x

    def linesearch(self, params, natural_gradient, advantages):
        # number of backtracks
        alpha = 0.5
        for k in range(10):
            params_new = params + alpha**k * natural_gradient

        return params_new