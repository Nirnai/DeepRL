import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from algorithms import BaseRL, OffPolicy, ActionValueFunction
from utils.policies import DeterministicPolicy
from utils.helper import soft_target_update

class TD3(BaseRL, OffPolicy):
    def __init__(self, env):
        super(TD3, self).__init__(env)
        self.name = "TD3"
        self.critic = ActionValueFunction(self.param.qvalue, self.device)
        self.actor = DeterministicPolicy(self.param.policy, self.device)
        self.actor_target = deepcopy(self.actor)
        self.steps = 0                    

    
    def act(self, state, deterministic=True):
       ######################################
        initial = False
        # if self.env.last_u is None:
        #     initial = True
        if self.env.timestep.step_type == 0:
            initial = True
        ######################################
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state).float().to(self.device))
        if deterministic is False:
            if self.param.POLICY_EXPLORATION_NOISE != 0:
                action += torch.randn(action.shape).to(self.device) * self.param.POLICY_EXPLORATION_NOISE 
                action = action.cpu().numpy()
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_state, reward, done, _ = self.env.step(action) 
        self.memory.push(state, action, reward, next_state, done, initial)
        self.steps += 1
        # if done:
        #     next_state = self.env.reset()
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData

        # Add noise on experience
        noise = torch.randn(batch.action.shape).to(self.device) * self.param.POLICY_UPDATE_NOISE
        noise = torch.clamp(noise, -self.param.POLICY_CLIP, self.param.POLICY_CLIP)
        next_action = self.actor_target(batch.next_state) + noise

        # Update Critic
        with torch.no_grad():
            q1_next, q2_next = self.critic.target(batch.next_state, next_action)
            q_target = batch.reward + self.param.GAMMA * torch.min(q1_next, q2_next)
        q1, q2 = self.critic(batch.state, batch.action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic.optimize(critic_loss)

        # Delayed Actor Update
        if self.steps % self.param.POLICY_UPDATE_FREQ == 0:
            action = self.actor(batch.state)
            Q, _ = self.critic(batch.state, action)
            actor_loss = -Q.mean()
            self.actor.optimize(actor_loss)
            soft_target_update(self.actor, self.actor_target, self.param.policy['TAU'])
        
        # Return Metrics
        metrics = dict()
        if self.steps % 5000 == 0:
            data = self.memory.replay()
            q1,_ = self.critic(data.state[(data.initial).type(torch.BoolTensor)], data.action[(data.initial).type(torch.BoolTensor)])
            metrics['value'] = q1.mean().item()
        # if len(next_state) > 0:
        #     next_action = self.actor(next_state)
        #     q1, q2 = self.critic(next_state, next_action)
        #     metrics['value'] = torch.min(q1,q2).mean().item()
        # metrics['entropy'] = self.actor.entropy(batch.state).sum().item()
        return metrics