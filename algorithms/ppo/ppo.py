import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
from algorithms import BaseRL, OnPolicy
from utils.policies import GaussianPolicy, BoundedGaussianPolicy
from utils.values_functions import ValueFunction
from copy import deepcopy
import gym
# from envs.vectorized import DummyVecEnv
from utils.memory import Buffer



class PPO(BaseRL, OnPolicy):
    def __init__(self, env, param=None):
        super(PPO, self).__init__(env, param=param)
        self.name = 'PPO'
        self.critic = ValueFunction(self.param.value , self.device)
        self.actor = GaussianPolicy(self.param.policy, self.device)
        self.steps = 0
        self.episode_steps = 0

        if self.param.LR_SCHEDULE:
            schedule = lambda epoch: 1 - epoch/(self.param.evaluation['total_timesteps'] // self.param.BATCH_SIZE)
        else:
            schedule = lambda epoch: 1
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor.optimizer, schedule)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic.optimizer, schedule)

    def act(self, state, deterministic=False):
        self.steps += 1
        with torch.no_grad():
            if self.steps < self.param.DELAYED_START:
                action = self.env.action_space.sample()
            else:
                s = torch.from_numpy(state).float().to(self.device)
                self.actor.eval()
                action = self.actor(s, deterministic=deterministic).cpu().numpy() 
                a = torch.from_numpy(action).float().to(self.device)
            next_state, reward, done, _ = self.env.step(action)
           
            if not deterministic:
                done_bool = float(done)
                self.critic.eval()
                s_ = np.stack([state, next_state])
                s_ = torch.from_numpy(s_).float().to(self.device)
                value, next_value = self.critic(s_)
                log_pi = self.actor.log_prob(s, a)
                self.memory.store(state, action, reward, next_state, done_bool, value, next_value, log_pi)
                if done:
                    self.memory.process_episode(maximum_entropy=self.param.MAX_ENTROPY) 
        return next_state, reward, done
    
    @OnPolicy.loop
    def learn(self):
        pg_norm = 0
        rollouts = self.onPolicyData
        if self.param.ADVANTAGE_NORMALIZATION:
            rollouts['advantages'] = (rollouts['advantages'] - rollouts['advantages'].mean()) / (rollouts['advantages'].std() + 1e-5) 
        for i in range(self.param.EPOCHS):
            generator = self.data_generator(rollouts)
            for mini_batch in generator:
                s, a, returns, old_values, old_log_probs, advantages = mini_batch
                # Critic Step
                self.critic.train()
                values = self.critic(s)
                if self.param.CLIPPED_VALUE:
                    critic_loss = self.clipped_value_loss(old_values, values, returns)
                else:
                    critic_loss = F.mse_loss(values, returns)
                self.critic.optimize(critic_loss)
                
                # Actor Step
                self.actor.train()
                log_probs = self.actor.log_prob(s,a)
                kl_div = (old_log_probs-log_probs).mean()   
                # Early Stopping            
                if self.param.EARLY_STOPPING and kl_div > self.param.MAX_KL_DIV:
                    print('Early stopping at epoch {} due to reaching max kl.'.format(i))
                    break
                actor_loss = self.clipped_policy_objective(old_log_probs, log_probs, advantages)
                actor_loss -= self.param.ENTROPY_COEFFICIENT * log_probs.mean()
                pg_norm += self.actor.optimize(actor_loss)
        self.critic_scheduler.step()
        self.actor_scheduler.step()
        # with torch.no_grad():
        #     pg = parameters_to_vector(torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True))
        #     total_stepsize += pg.norm()
                
                

        metrics = dict()
        with torch.no_grad():
            metrics['explained_variance'] = (1 - (rollouts['returns_mc'] - rollouts['values']).pow(2).sum()/(rollouts['returns_mc']-rollouts['returns_mc'].mean() + 1e-5).pow(2).sum()).item()
            metrics['entropy'] = self.actor.entropy(rollouts['states']).mean().item()
            metrics['kl'] = kl_div.item()
            metrics['pg_norm'] = pg_norm
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
        loss = (val - ret).pow(2)
        clipped_loss = ((old_val + torch.clamp(val - old_val, -self.param.CLIP, self.param.CLIP)) - ret).pow(2)
        return torch.min(loss, clipped_loss).mean()
        # clipped_val = old_val + (val - old_val).clamp(-self.param.CLIP, self.param.CLIP)
        # loss = (val - ret).pow(2)
        # clipped_loss = (clipped_val - ret).pow(2)
        # return torch.max(loss, clipped_loss).mean()

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
        self.var = np.ones(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = (self.mean_diff/self.n).clip(min=1e-5)

    def normalize(self, inputs):
        if type(inputs) == torch.Tensor:
            obs_std = torch.sqrt(torch.from_numpy(self.var).float())
            obs_mean = torch.from_numpy(self.mean).float()
        else:
            obs_std = np.sqrt(self.var)
            obs_mean = self.mean
        return (inputs - obs_mean)/obs_std

class MinMaxNormalizer():
    def __init__(self, num_inputs):
        self.min = None
        self.max = None

    def observe(self, x):
        if self.min is None:
            self.min = x
        if self.max is None:
            self.max = x
        self.min = np.minimum(x, self.min)
        self.max = np.maximum(x, self.max)
    

    def normalize(self, inputs):
        if type(inputs) == torch.Tensor:
            x_min = torch.from_numpy(self.min).float()
            x_max = torch.from_numpy(self.max).float()
        else:
            x_min = self.min
            x_max = self.max
        return 2*(inputs-x_min)/(x_max - x_min + 1e-5)