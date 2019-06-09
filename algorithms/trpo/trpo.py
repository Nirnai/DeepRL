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
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from algorithms import BaseRL, HyperParameter
from utils.models import Policy, Value
from utils.memory import RolloutBuffer
from utils.env import getEnvInfo


class TRPO(BaseRL):
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
            advantages = (advantages - advantages.mean()) / advantages.std()  

            self.critic_optim.zero_grad()
            value_loss = advantages.pow(2.).mean()
            value_loss.backward()
            self.critic_optim.step() 

            advantages = advantages.detach()
            p = self.actor(states)
            log_probs = p.log_prob(actions).squeeze()
            loss = (log_probs * advantages).mean()
            pg = parameters_to_vector(torch.autograd.grad(loss, self.actor.parameters()))

            def get_kl(model):
                with torch.no_grad():
                    p_old = self.actor(states)
                p_new = model(states)
                d_kl = kl_divergence(p_old, p_new).mean()
                return d_kl
            
            def Hx(x):
                d_kl = get_kl(self.actor)    
                grads = torch.autograd.grad(d_kl, self.actor.parameters(), create_graph=True)
                grads = parameters_to_vector(grads)
                Jx = torch.sum(grads * x)
                Hx = torch.autograd.grad(Jx, self.actor.parameters())
                Hx = parameters_to_vector(Hx)
                return Hx + 0.1 * x

            stepdir = self.conjugate_gradient(Hx, pg, self.param.NUM_CG_ITER)  
            stepsize = (2 * self.param.DELTA) / torch.dot(stepdir,Hx(stepdir))
            npg = torch.sqrt(stepsize) * stepdir
            params_new = self.linesearch(npg, pg, get_kl)
            vector_to_parameters(params_new, self.actor.parameters())
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


    def conjugate_gradient(self, A, b, n):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = torch.dot(r,r)
        for i in range(n):
            if callable(A):
                Ap = A(p)
            else:
                Ap = torch.matmul(A,p)
            alpha = rs / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_next = torch.dot(r, r)
            betta = rs_next / rs
            p = r + betta * p
            rs = rs_next
            if rs < 1e-10:
                break
        return x

    def linesearch(self, npg, pg, get_kl):
        params_curr = self.actor.get_params()
        for k in range(self.param.NUM_BACKTRACK):
            params_new = params_curr + self.param.ALPHA**k * npg 
            model_new = deepcopy(self.actor)
            vector_to_parameters(params_new, model_new.parameters())
            param_diff = params_new - params_curr
            surr_loss = torch.dot(pg,param_diff)
            kl_div = get_kl(model_new)
            if surr_loss >= 0 and kl_div <= self.param.DELTA:
                params_curr = params_new
                break
        return params_curr