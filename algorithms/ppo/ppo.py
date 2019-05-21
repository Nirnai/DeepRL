import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algorithms.utils import ActorCritic, Policy, Value
from algorithms.utils import getEnvInfo
from collections import namedtuple

Transition = namedtuple('Transition', ('states', 'actions' ,'rewards', 'next_states', 'mask'))


class PPO():
    def __init__(self, env, param):
        self.name = "A2C"
        self.env = env
        self.param = param
        self.rng = random.Random()

        self.state_dim, self.action_dim, self.action_space = getEnvInfo(env)
        self.param.ACTOR_ARCHITECTURE[0] = self.state_dim
        self.param.ACTOR_ARCHITECTURE[-1] = self.action_dim

        architecture = self.param.ACTOR_ARCHITECTURE
        activation = self.param.ACTIVATION

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.actor = Policy(architecture, activation, action_space=self.action_space)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.param.LEARNING_RATE)

        self.critic = Value(architecture, activation)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.param.LEARNING_RATE)

        ##############################################################################

        self.rollout = []

        self.steps = 0
        self.done = False
    
    def act(self, state):
        ''' '''
        policy = self.actor(torch.from_numpy(state).float())
        action = policy.sample() 
        next_state, reward, done, _ = self.env.step(action.numpy()) 
        self.rollout.append(Transition(state, action, reward, next_state, (1-done)))
        self.steps += 1
        # self.done = done  
        return next_state, reward, done


    def learn(self):
        if len(self.rollout) == self.param.BATCH_SIZE:
            for _ in range(4):
                rollout = Transition(*zip(*self.rollout))
                states = torch.Tensor(rollout.states)
                actions = torch.stack(rollout.actions).squeeze()
                rewards = torch.Tensor(rollout.rewards)
                next_states = torch.Tensor(rollout.next_states)
                mask = torch.Tensor(rollout.mask)

                advantages, values, returns= self.generaized_advantage_estimation(states, next_states[-1], rewards, mask)

                # Optimizer Critic
                self.critic_optim.zero_grad()
                # value_loss = (values - returns).pow(2.).mean()
                value_loss = nn.functional.mse_loss(values, returns)
                value_loss.backward()
                self.critic_optim.step()

                curr_policy = self.actor(states)
                curr_log_probs = curr_policy.log_prob(actions)
                old_policy = self.actor(states, old=True)
                old_log_probs = old_policy.log_prob(actions)
                
                # Optimize Policy
                self.actor.backup()
                self.actor_optim.zero_grad()
                ratio = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratio*advantages
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss.backward()
                self.actor_optim.step()



            # entropy = torch.stack(rollout.entropy).sum()
            
            # value_targets = self.bootstrap_target_estimations(next_states, rewards, mask)
            # advantages = (value_targets - self.critic(states).squeeze()).detach()
            # Monte Carlo Advantages
            # advantages = self.monte_carlo_advantage_estimation(rewards, values, mask)
            # General Advantages
            # advantages = self.generaized_advantage_estimation(states, rewards, mask)
            # for _ in range(epochs):
            #     for batch_index in BatchSampler(SubsetRandomSampler(range(self.param.BUFFER_SIZE)), self.param.BATCH_SIZE, False):
            #         #batch_index = self.rng.sample(range(self.param.STEPS), self.param.BATCH_SIZE)
            #         state_batch = states[batch_index]
            #         action_batch = actions[batch_index]
            #         log_probs_old_batch = log_probs_old[batch_index].detach()
            #         advantage_batch = advantages[batch_index]

            #         policy = self.actor(state_batch)
            #         # entropy_batch = policy.entropy().mean()
            #         log_probs_new_batch = policy.log_prob(action_batch).squeeze()

            #         ratio = (log_probs_new_batch - log_probs_old_batch).exp()
            #         surr1 = ratio * advantage_batch
            #         surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_batch.detach()

            #         actor_loss  = - torch.min(surr1, surr2).mean()

            #         critic_loss = F.smooth_l1_loss(self.critic(states), value_targets)
            #         # critic_loss = advantage_batch.pow(2).mean()
            #         # loss = actor_loss + 0.5 * critic_loss  - 0.001 * entropy_batch

            #         self.actor_optimizer.zero_grad()
            #         actor_loss.backward()
            #         self.actor_optimizer.step()

            #         self.critic_optimizer.zero_grad()
            #         critic_loss.backward()
            #         self.critic_optimizer.step()

            del self.rollout[:]

    
    def bootstrap_target_estimations(self, next_states, rewards, mask):
        pass
        # with torch.no_grad():
        #     targets = rewards + self.param.GAMMA * self.critic(next_states).squeeze()
        # return targets


    def generaized_advantage_estimation(self, states, next_state ,rewards, mask):
        '''  Generaized Advantage Estimation '''

        # TODO: mask has no termination!!!!!!!

        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_value = self.critic(next_state)

        values = torch.cat((values, next_value))
        returns = [0] * (len(rewards)+1)
        advantages = [0] * (len(rewards)+1)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.param.GAMMA * values[t+1] * mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * mask[t] * advantages[t+1]
            returns[t] = rewards[t] + self.param.GAMMA * returns[t+1] * mask[t]
        return torch.stack(advantages).detach(), values[:-1] , torch.stack(returns[:-1]).squeeze()


    def monte_carlo_advantage_estimation(self, states, rewards, mask):
        pass
        # with torch.no_grad():
        #     values = self.critic(states)
        # Q = 0
        # returns = []
        # for t in reversed(range(len(rewards))):
        #     Q = rewards[t] + self.param.GAMMA * Q * mask[t]
        #     returns.insert(0, Q)   
        # advantages = torch.Tensor(returns) - values * mask
        # return advantages


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)
