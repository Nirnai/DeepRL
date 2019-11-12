import numpy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from algorithms import BaseRL, HyperParameter, OffPolicy
from utils.policies import BoundedGaussianPolicy
from utils.values_functions import ActionValueFunction

class SAC(BaseRL, OffPolicy):
    def __init__(self, env, param=None):
        super(SAC, self).__init__(env, param=param)
        self.name = 'SAC'
        self.critic = ActionValueFunction(self.param.qvalue, self.device)
        self.actor = BoundedGaussianPolicy(self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy()
            action *= self.env.action_space.high
        next_state, reward, done, _ = self.env.step(action)
        if done:
            next_state = self.env.reset() 
        self.memory.store(state, action, reward, next_state, done)
        self.steps += 1
        return next_state, reward, done

    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData
        # Update Critic
        q1, q2 = self.critic(batch['states'], batch['actions'])
        with torch.no_grad():
            new_action_next = self.actor(batch['next_states'])
            log_prob_next = self.actor.log_prob(batch['next_states'], new_action_next)
            q1_next, q2_next = self.critic.target(batch['next_states'], new_action_next)
            q_target = batch['rewards'] + self.param.GAMMA * torch.min(q1_next, q2_next) - self.param.ALPHA * log_prob_next
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic.optimize(critic_loss)

        # Update Actor
        if self.steps % self.param.POLICY_UPDATE_FREQ == 0:
            new_action, log_prob = self.actor.rsample(batch['states'])  
            q1, q2 = self.critic(batch['states'], new_action)
            actor_loss = (self.param.ALPHA * log_prob - torch.min(q1,q2)).mean()
            self.actor.optimize(actor_loss)
        
        # Return Metrics
        metrics = dict()
        if self.steps % 5000 == 0:
            # data = self.memory.sample(5000)
            # q1,_ = self.critic(data['states'], data['actions'])
            # metrics['value'] = q1.mean().item()
            metrics['entropy'] = (-self.actor.log_prob(batch['states'], batch['actions'])).mean().item()
        return metrics