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
        # self.init_policy()
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
            s = torch.from_numpy(state).float().to(self.device)
            if self.steps < self.param.DELAYED_START:
                action = self.env.action_space.sample()
            else:
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
                # if done:
                #     self.memory.process_episode(maximum_entropy=self.param.MAX_ENTROPY) 
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
                for _ in range(self.param.VALUE_EPOCHS):
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
                    # print('Early stopping at epoch {} due to reaching max kl.'.format(i))
                    break
                actor_loss = self.clipped_policy_objective(old_log_probs, log_probs, advantages)
                actor_loss -= self.param.ENTROPY_COEFFICIENT * log_probs.mean()
                pg_norm += self.actor.optimize(actor_loss)
        self.critic_scheduler.step()
        self.actor_scheduler.step()
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


    def init_policy(self):
        init_mean = 0.5 * (self.env.action_space.low + self.env.action_space.high) * np.ones(self.env.action_space.shape[0])
        init_std =  0.5 * (self.env.action_space.high - self.env.action_space.low) * np.ones(self.env.action_space.shape[0])
        for batchIdx in range(2000):
            states = np.random.normal(0,1,size=[256, self.env.observation_space.shape[0]])
            action_mean = self.actor(torch.from_numpy(states).float(), deterministic=True)
            action_std = torch.exp(self.actor.log_std)
            mean_target = torch.ones_like(action_mean).float() * torch.from_numpy(init_mean).float()
            std_target = torch.from_numpy(init_std).float()
            loss = torch.nn.functional.mse_loss(action_mean, mean_target) + \
                torch.nn.functional.mse_loss(action_std, std_target)
            # print(loss)
            self.actor.optimize(loss)