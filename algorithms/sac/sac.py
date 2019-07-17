import os
import inspect
import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from algorithms import BaseRL, HyperParameter, OffPolicy, QModel
from utils.policies import BoundedGaussianPolicy

class SAC(BaseRL, OffPolicy):
    def __init__(self, env):
        super(SAC, self).__init__(env)
        self.name = 'SAC'
        self.critic = QModel(self.param)
        self.actor = BoundedGaussianPolicy(self.param.ARCHITECTURE, 
                                           self.param.ACTIVATION, 
                                           self.param.LEARNING_RATE)
        self.steps = 0

    def act(self, state, deterministic=False):
        action = self.actor(torch.from_numpy(state).float(), deterministic=deterministic)
        next_state, reward, done, _ = self.env.step(action.numpy()) 
        self._memory.push(state, action, reward, next_state, done)
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done

    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData

        # Update Critic
        q1, q2 = self.critic(batch.state, batch.action)
        new_action_next, log_prob_next = self.actor.rsample(batch.next_state)
        with torch.no_grad():
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
        metrics['entropy'] = self.actor.entropy(batch.state).sum().item()
        return metrics

# class SAC(BaseRL):
#     def __init__(self, env):
#         super(SAC,self).__init__(env)
#         self.name = "SAC"

#         parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
#         self.param = HyperParameter(parameters_file)
#         self.param.ARCHITECTURE[ 0 ] = self.state_dim
#         self.param.ARCHITECTURE[-1 ] = self.action_dim
#         architecture = self.param.ARCHITECTURE
#         activation = self.param.ACTIVATION

#         self.actor = Policy(architecture, activation, action_space=self.action_space)
#         self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
#         self.q1 = QValue(architecture, activation) 
#         self.q2 = QValue(architecture, activation) 
#         self.q1_target = deepcopy(self.q1)
#         self.q2_target = deepcopy(self.q2)
#         self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.param.LEARNING_RATE)
        
#         self.replay_buffer = Memory(int(self.param.MEMORY_SIZE), self.rng)
#         self.steps = 0

    
#     def act(self, state, exploit=False):
#         with torch.no_grad():
#             policy = self.actor.policy(torch.from_numpy(state).float())
#             if exploit:
#                 action = torch.tanh(policy.mean.detach())
#             else:
#                 action = torch.tanh(policy.sample())    
#         next_state, reward, done, _ = self.env.step(action.numpy()) 
#         self.replay_buffer.push(state, action, reward, next_state, done)
#         self.steps += 1
#         if done:
#             next_state = self.env.reset()
#         return next_state, reward, done
    

    
#     def learn(self):
#         if len(self.replay_buffer) > self.param.BATCH_SIZE:
#             transitions = self.replay_buffer.sample(self.param.BATCH_SIZE)
#             state_batch = torch.Tensor(transitions.state)
#             next_state_batch = torch.Tensor(transitions.next_state)
#             action_batch = torch.Tensor(transitions.action)
#             reward_batch = torch.Tensor(transitions.reward)
#             mask_batch = torch.Tensor(transitions.mask)

#             # Resample from policy and bound output
#             policy = self.actor.policy(state_batch)
#             action = policy.rsample()
#             log_probs = (policy.log_prob(action) - torch.log1p(-torch.tanh(action).pow(2) + 1e-6)).squeeze()
#             actions_bounded = torch.tanh(action)
#             Q = torch.min(self.q1(state_batch, actions_bounded) ,self.q2(state_batch, actions_bounded))
#             policy_loss = (self.param.ALPHA * log_probs - Q).mean()

#             Q1 = self.q1(state_batch, action_batch)
#             Q2 = self.q2(state_batch, action_batch)
#             policy_next = self.actor.policy(next_state_batch)
#             action_next = policy_next.rsample()
#             log_probs_next = (policy_next.log_prob(action_next) - torch.log1p(-torch.tanh(action_next).pow(2) + 1e-6)).squeeze()
#             actions_next_bounded = torch.tanh(action_next)
           
#             with torch.no_grad():
#                 Q1_next = self.q1_target(next_state_batch, actions_next_bounded)
#                 Q2_next = self.q2_target(next_state_batch, actions_next_bounded)
#                 Q_target = reward_batch + self.param.GAMMA * mask_batch * torch.min(Q1_next, Q2_next) - self.param.ALPHA * log_probs_next

#             Q_loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)
#             self.q_optim.zero_grad()
#             Q_loss.backward()
#             self.q_optim.step()

#             self.actor_optim.zero_grad()
#             policy_loss.backward()
#             self.actor_optim.step()

#             soft_target_update(self.q1, self.q1_target, self.param.TAU)
#             soft_target_update(self.q2, self.q2_target, self.param.TAU)

#             metrics = dict()
#             metrics['loss'] = Q_loss.item()
#             metrics['entropy'] = self.actor.entropy(state_batch).sum().item()
#             return metrics




# class SAC(BaseRL):
#     def __init__(self, env):
#         super(SAC,self).__init__(env)
#         self.name = "SAC"

#         parameters_file = os.path.dirname(inspect.getfile(self.__class__)) + '/parameters.json'
#         self.param = HyperParameter(parameters_file)
#         self.param.ARCHITECTURE[ 0 ] = self.state_dim
#         self.param.ARCHITECTURE[-1 ] = self.action_dim
#         architecture = self.param.ARCHITECTURE
#         activation = self.param.ACTIVATION

#         self.actor = Policy(architecture, activation, action_space=self.action_space)
#         self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)
#         self.vcritic = Value(architecture, activation)
#         self.vcritic_target = deepcopy(self.vcritic)
#         self.vcritic_optim = optim.Adam(self.vcritic.parameters(), lr=self.param.LEARNING_RATE)
#         self.qcritic1 = QValue(architecture, activation) 
#         self.qcritic2 = QValue(architecture, activation) 
#         self.qritics_optim = optim.Adam(list(self.qcritic1.parameters()) + list(self.qcritic2.parameters()), lr=self.param.LEARNING_RATE)
        
#         self.replay_buffer = Memory(int(self.param.MEMORY_SIZE), self.rng)
#         self.steps = 0

    
#     def act(self, state, exploit=False):
#         with torch.no_grad():
#             policy = self.actor.policy(torch.from_numpy(state).float())
#             if exploit:
#                 action = torch.tanh(policy.mean.detach())
#             else:
#                 action = torch.tanh(policy.sample())    
#         next_state, reward, done, _ = self.env.step(action.numpy()) 
#         self.replay_buffer.push(state, action, reward, next_state, done)
#         self.steps += 1
#         if done:
#             next_state = self.env.reset()
#         return next_state, reward, done
    

    
#     def learn(self):
#         if len(self.replay_buffer) > self.param.BATCH_SIZE:
#             transitions = self.replay_buffer.sample(self.param.BATCH_SIZE)
#             state_batch = torch.Tensor(transitions.state)
#             next_state_batch = torch.Tensor(transitions.next_state)
#             action_batch = torch.Tensor(transitions.action)
#             reward_batch = torch.Tensor(transitions.reward)
#             mask_batch = torch.Tensor(transitions.mask)

#             # Resample from policy and bound output
#             policy = self.actor.policy(state_batch)
#             action = policy.rsample()
#             log_probs = (policy.log_prob(action) - torch.log1p(-torch.tanh(action).pow(2) + 1e-6)).squeeze()
#             actions_bounded = torch.tanh(action)

#             Q1 = self.qcritic1(state_batch, actions_bounded.detach())
#             Q2 = self.qcritic2(state_batch, actions_bounded.detach())
            
#             V_target = torch.min(Q1, Q2) - self.param.ALPHA * log_probs.detach()
#             Q_target = reward_batch + self.param.GAMMA * mask_batch * self.vcritic_target(next_state_batch)

#             V_loss = (self.vcritic(state_batch).squeeze() - V_target).pow(2).mean()
#             self.vcritic_optim.zero_grad()
#             V_loss.backward()
#             self.vcritic_optim.step()

#             Q_loss = (self.qcritic1(state_batch, action_batch) - Q_target).pow(2).mean() + \
#                     (self.qcritic2(state_batch, action_batch) - Q_target).pow(2).mean()
#             self.qritics_optim.zero_grad()
#             Q_loss.backward()
#             self.qritics_optim.step()

#             policy_loss = (self.param.ALPHA * log_probs - self.qcritic1(state_batch, actions_bounded)).mean()
#             self.actor_optim.zero_grad()
#             policy_loss.backward()
#             self.actor_optim.step()

#             soft_target_update(self.vcritic, self.vcritic_target, self.param.TAU)

#             metrics = dict()
#             metrics['loss'] = Q_loss.item()
#             metrics['entropy'] = self.actor.entropy(state_batch).sum().item()
#             return metrics