import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from algorithms import BaseRL, OffPolicy, QModel
from utils.policies import DeterministicPolicy
from utils.helper import soft_target_update

class TD3(BaseRL, OffPolicy):
    def __init__(self, env):
        super(TD3, self).__init__(env)
        self.name = "TD3"
        self.critic = QModel(self.param)
        self.actor = DeterministicPolicy(self.param.ARCHITECTURE, 
                                         self.param.ACTIVATION,
                                         self.param.LEARNING_RATE)
        self.actor_target = deepcopy(self.actor)
        self.steps = 0                    

    
    def act(self, state, noise=0.1):
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state).float())
        if noise != 0:
            action += torch.randn(action.shape) * noise 
            action = torch.clamp(action, self.env.action_space.low.item(), self.env.action_space.high.item())
        next_state, reward, done, _ = self.env.step(action.numpy()) 
        self._memory.push(state, action, reward, next_state, done)
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData

        noise = torch.randn(batch.action.shape) * self.param.POLICY_NOISE
        noise = torch.clamp(noise, -self.param.POLICY_CLIP, self.param.POLICY_CLIP)
        next_action = self.actor_target(batch.next_state) + noise

        # Update Critic
        with torch.no_grad():
            q1_next, q2_next = self.critic.target(batch.next_state, next_action)
            q_target = batch.reward + self.param.GAMMA * batch.mask * torch.min(q1_next, q2_next)
        q1, q2 = self.critic(batch.state, batch.action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic.optimize(critic_loss)

        # Delayed Actor Update
        if (self.steps - self.param.BATCH_SIZE) % self.param.POLICY_UPDATE_FREQ == 0:
            action = self.actor(batch.state)
            Q, _ = self.critic(batch.state, action)
            actor_loss = -Q.mean()
            self.actor.optimize(actor_loss)
            soft_target_update(self.actor, self.actor_target, self.param.TAU)