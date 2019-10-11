import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
from algorithms import BaseRL, OnPolicy, ValueFunction
from utils.policies import GaussianPolicy, BoundedGaussianPolicy
from copy import deepcopy
import gym
from envs.vectorized import DummyVecEnv
from utils.memory import Buffer



class PPO(BaseRL, OnPolicy):
    def __init__(self, env):
        super(PPO, self).__init__(env)
        self.name = 'PPO'
        self.critic = ValueFunction(self.param.value , self.device)
        self.actor = GaussianPolicy(self.param.policy, self.device)
        self.steps = 0
        self.episode_steps = 0

        if self.param.LR_SCHEDULE:
            schedule = lambda epoch: 1 - epoch/(self.param.evaluation['total_timesteps'] // self.param.BATCH_SIZE)
        else:
            schedule = lambda epoch: 1
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor._optim, schedule)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic.V_optim, schedule)

        self.state_normalizer = Normalizer(self.env.observation_space.shape[0])

    def act(self, state, deterministic=False):
        self.steps += 1
        self.state_normalizer.observe(state)
        with torch.no_grad():
            # state = self.state_normalizer.normalize(state)
            if self.steps < self.param.DELAYED_START:
                action = self.env.action_space.sample()
            else:
                action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy() 
            next_state, reward, done, _ = self.env.step(action)
            # next_state_norm = self.state_normalizer.normalize(next_state)
           
            if not deterministic:
                done_bool = float(done) if self.episode_steps < self.env._max_episode_steps else 0
                value = self.critic(torch.from_numpy(state).float().to(self.device))
                next_value = self.critic(torch.from_numpy(next_state).float().to(self.device))
                log_pi = self.actor.log_prob(torch.from_numpy(state).float().to(self.device), 
                                            torch.from_numpy(action).float().to(self.device))
                self.memory.store(state, action, reward, next_state, done_bool, value, next_value, log_pi)
                if done:
                    self.memory.process_episode() 
        return next_state, reward, done
    
    @OnPolicy.loop
    def learn(self):
        rollouts = self.onPolicyData
        rollouts['advantages'] = (rollouts['advantages'] - rollouts['advantages'].mean()) / (rollouts['advantages'].std() + 1e-5) 
        for _ in range(self.param.EPOCHS):
            generator = self.data_generator(rollouts)
            for mini_batch in generator:
                s, a, returns, old_values, old_log_probs, advantages = mini_batch
                # Critic Step
                values = self.critic(s)
                if self.param.CLIPPED_VALUE:
                    critic_loss = self.clipped_value_loss(old_values, values, returns)
                else:
                    critic_loss = F.mse_loss(values, returns)
                # clip_grad_norm_(self.critic.parameters(), self.param.MAX_GRAD_NORM)
                self.critic.optimize(critic_loss)
                
                # Actor Step
                log_probs = self.actor.log_prob(s,a)
                kl_div = (old_log_probs-log_probs).mean()   
                # Early Stopping            
                if self.param.EARLY_STOPPING and kl_div > self.param.MAX_KL_DIV:
                    break
                actor_loss = self.clipped_policy_objective(old_log_probs, log_probs, advantages)
                actor_loss -= self.param.ENTROPY_COEFFICIENT * log_probs.mean()
                # clip_grad_norm_(self.actor.parameters(), self.param.MAX_GRAD_NORM)
                self.actor.optimize(actor_loss)
        self.critic_scheduler.step()
        self.actor_scheduler.step()
        # with torch.no_grad():
        #     pg = parameters_to_vector(torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True))
        #     total_stepsize += pg.norm()
                
                

        metrics = dict()
        with torch.no_grad():
            # metrics['learning_rate'] = [param_group['lr'] for param_group in self.actor._optim.param_groups][-1]
            # metrics['state_mean'] = self.state_normalizer.mean.mean()
            # metrics['state_var'] = self.state_normalizer.var.mean()
            # metrics['value'] = values.mean().item()
            # metrics['target'] = returns.mean().item()
            metrics['explained_variance'] = (1 - (rollouts['returns_mc'] - rollouts['values']).pow(2).sum()/(rollouts['returns_mc']-rollouts['returns_mc'].mean()).pow(2).sum()).item()
            # metrics['entropy'] = self.actor.entropy(rollouts['states']).mean().item()
            # metrics['kl'] = total_kl.item()
            # metrics['stepsize'] = total_stepsize.item()
        return metrics

    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def clipped_policy_objective(self, old_log_pi, log_pi, adv):
        ratio = torch.exp(log_pi - old_log_pi)
        loss = ratio * adv
        clipped_loss = torch.clamp(ratio, 1 - self.param.CLIP,  1 + self.param.CLIP) * adv
        return -torch.min(loss, clipped_loss).mean()

    def clipped_value_loss(self, old_val, val, ret):
        clipped_val = old_val + (val - old_val).clamp(-self.param.CLIP, self.param.CLIP)
        loss = (val - ret).pow(2)
        clipped_loss = (clipped_val - ret).pow(2)
        return torch.max(loss, clipped_loss).mean()
        
    # def importance_weights(self, states, actions): 
    #     with torch.no_grad():
    #         old_log_probs = self.actor_old.log_prob(states, actions)
    #     curr_log_probs = self.actor.log_prob(states, actions)
    #     kl = (old_log_probs-curr_log_probs).mean()
    #     ratio = torch.exp(curr_log_probs - old_log_probs)
    #     return ratio, kl

    def data_generator(self, rollouts):
        if self.param.NUM_MINI_BATCHES > 0:
            mini_batch_size = self.param.BATCH_SIZE // self.param.NUM_MINI_BATCHES
            random_sampler = SubsetRandomSampler(range(self.param.BATCH_SIZE))
            batch_sampler = BatchSampler(random_sampler, mini_batch_size, drop_last=True)
            for indices in batch_sampler:
                s = rollouts['states'][indices]
                a = rollouts['actions'][indices]
                ret = rollouts['returns_gae'][indices]
                val = rollouts['values'][indices]
                pi = rollouts['log_probs'][indices]
                adv = rollouts['advantages'][indices]
                yield s, a, ret, val, pi, adv
        else:
            s = rollouts['states']
            a = rollouts['actions']
            ret = rollouts['returns_gae']
            val = rollouts['values']
            pi = rollouts['log_probs']
            adv = rollouts['advantages']
            yield s, a, ret, val, pi, adv


class Normalizer():
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.mean_diff/self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean)/obs_std