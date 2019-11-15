import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from algorithms import BaseRL, OffPolicy
from utils.policies import DeterministicPolicy
from utils.values_functions import ActionValueFunction
from utils.helper import soft_target_update

class TD3(BaseRL, OffPolicy):
    def __init__(self, env, param=None):
        super(TD3, self).__init__(env, param=param)
        self.name = "TD3"
        self.critic = ActionValueFunction(self.param.qvalue, self.device)
        self.actor = DeterministicPolicy(self.param.policy, self.device)
        self.actor_target = deepcopy(self.actor)
        self.steps = 0                    

    
    def act(self, state, deterministic=False):
        self.steps += 1
        if self.steps < self.param.DELAYED_START:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action = self.actor(torch.from_numpy(state).float().to(self.device)) * torch.from_numpy(self.env.action_space.high)
            if deterministic is False:
                if self.param.POLICY_EXPLORATION_NOISE != 0:
                    action += torch.randn(action.shape).to(self.device) * self.param.POLICY_EXPLORATION_NOISE 
            action = action.cpu().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)            
        next_state, reward, done, _ = self.env.step(action) 
        self.memory.store(state, action, reward, next_state, done)
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData
        # Add noise on experience
        noise = torch.randn(batch['actions'].shape).to(self.device) * self.param.POLICY_UPDATE_NOISE
        noise = torch.clamp(noise, -self.param.POLICY_CLIP, self.param.POLICY_CLIP)
        next_action = self.actor_target(batch['next_states']) + noise
        # Update Critic
        with torch.no_grad():
            q1_next, q2_next = self.critic.target(batch['next_states'], next_action)
            q_target = batch['rewards'] + self.param.GAMMA * torch.min(q1_next, q2_next)
        q1, q2 = self.critic(batch['states'], batch['actions'])
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic.optimize(critic_loss)

        # Delayed Actor Update
        if self.steps % self.param.POLICY_UPDATE_FREQ == 0:
            action = self.actor(batch['states'])
            Q, _ = self.critic(batch['states'], action)
            actor_loss = -Q.mean()
            self.actor.optimize(actor_loss)
            soft_target_update(self.actor, self.actor_target, self.param.policy['TAU'])
        
        # Return Metrics
        metrics = dict()
        # if self.steps % 1000 == 0:
        #     data = self.memory.sample(5000)
        #     q1,_ = self.critic(data['states'], data['actions'])
        #     metrics['values'] = q1.mean().item()
        return metrics