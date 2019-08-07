import numpy
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from algorithms import BaseRL, HyperParameter, OffPolicy, ActionValueFunction
from utils.policies import BoundedGaussianPolicy

class SAC(BaseRL, OffPolicy):
    def __init__(self, env):
        super(SAC, self).__init__(env)
        self.name = 'SAC'
        self.critic = ActionValueFunction(self.param.qvalue, self.device)
        self.actor = BoundedGaussianPolicy(self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)
        if done:
            next_state = self.env.reset() 
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        return next_state, reward, done

    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData
        # Update Critic
        q1, q2 = self.critic(batch.state, batch.action)
        with torch.no_grad():
            new_action_next, log_prob_next = self.actor.rsample(batch.next_state)
            q1_next, q2_next = self.critic.target(batch.next_state, new_action_next)
            q_target = batch.reward + self.param.GAMMA * batch.mask * torch.min(q1_next, q2_next) - self.param.ALPHA * log_prob_next
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic.optimize(critic_loss)

        # Update Actor
        new_action, log_prob = self.actor.rsample(batch.state)  
        q1, q2 = self.critic(batch.state, new_action)
        actor_loss = (self.param.ALPHA * log_prob - torch.min(q1,q2)).mean()
        self.actor.optimize(actor_loss)
        
        # Return Metrics
        metrics = dict()
        # data = self._memory.replay()
        # terminals = (1-data.mask).type(torch.BoolTensor)
        # next_state = data.next_state[terminals]
        # if len(next_state) > 0:
        #     next_action = self.actor(next_state)
        #     q1, q2 = self.critic(next_state, next_action)
        #     metrics['value'] = torch.min(q1,q2).mean().item()
        # metrics['entropy'] = self.actor.entropy(batch.state).sum().item()
        return metrics