import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal
from copy import deepcopy
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from algorithms import BaseRL, OnPolicy
from utils.policies import GaussianPolicy, BoundedGaussianPolicy
from utils.values_functions import ValueFunction


class TRPO(BaseRL, OnPolicy):
    def __init__(self, env, param=None):
        super(TRPO,self).__init__(env, param=param)
        self.name = "TRPO"
        self.critic = ValueFunction(self.param.value, self.device)
        self.actor = GaussianPolicy(self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        self.steps += 1
        with torch.no_grad():
            if self.steps < self.param.DELAYED_START:
                action = self.env.action_space.sample()
            else:
                self.actor.eval()
                action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy() 
            next_state, reward, done, _ = self.env.step(action)
            if not deterministic:
                done_bool = float(done) #if self.episode_steps < self.env._max_episode_steps else 0
                self.critic.eval()
                value, next_value = self.critic(torch.from_numpy(np.stack([state, next_state])).float().to(self.device))
                # value = self.critic(torch.from_numpy(state).float().to(self.device))
                # next_value = self.critic(torch.from_numpy(next_state).float().to(self.device))
                
                log_pi = self.actor.log_prob(torch.from_numpy(state).float().to(self.device), 
                                            torch.from_numpy(action).float().to(self.device))
                self.memory.store(state, action, reward, next_state, done_bool, value, next_value, log_pi)
                if done:
                    self.memory.process_episode(maximum_entropy=self.param.MAX_ENTROPY) 
        return next_state, reward, done
    

    @OnPolicy.loop
    def learn(self):
        rollouts = self.onPolicyData
        returns = rollouts['returns_gae']
        if self.param.ADVANTAGE_NORMALIZATION:
            rollouts['advantages'] = (rollouts['advantages'] - rollouts['advantages'].mean()) / (rollouts['advantages'].std() + 1e-5) 
        for _ in range(self.param.EPOCHS):
            # Compute Advantages
            for _ in range(self.param.VALUE_EPOCHS):
                # Update Critic
                values = self.critic(rollouts['states'])
                critic_loss = F.mse_loss(values, returns)
                self.critic.optimize(critic_loss)
            # Update Actor
            old_log_probs = self.actor.log_prob(rollouts['states'], rollouts['actions'])
            pg = self.policy_gradient(rollouts)
            npg = self.natural_gradient(pg, rollouts)
            parameters, pg_norm = self.linesearch(npg, pg, rollouts)
            self.optimize_actor(parameters)
            log_probs = self.actor.log_prob(rollouts['states'], rollouts['actions'])

        metrics = dict()
        with torch.no_grad():
            metrics['explained_variance'] = (1 - (rollouts['returns_mc'] - rollouts['values']).pow(2).sum()/(rollouts['returns_mc']-rollouts['returns_mc'].mean()).pow(2).sum()).item()
            metrics['entropy'] = self.actor.entropy(rollouts['states']).mean().item()
            metrics['kl'] = (old_log_probs-log_probs).mean()
            metrics['pg_norm'] = pg_norm
        return metrics

    ################################################################
    ########################## Utilities ###########################
    ################################################################
    def optimize_actor(self, new_parameters):
        vector_to_parameters(new_parameters, self.actor.parameters())

    def policy_gradient(self, rollouts):
        log_probs = self.actor.log_prob(rollouts['states'], rollouts['actions'])
        pg_objective = (log_probs * rollouts['advantages']).mean()
        pg_objective -= self.param.ENTROPY_COEFFICIENT * rollouts['log_probs'].mean()
        return parameters_to_vector(torch.autograd.grad(pg_objective, self.actor.parameters()))

    def natural_gradient(self, pg, rollouts):
        def Hx(x):
            ''' Computes the Hessian-Vector product for the KL-Divergance '''
            d_kl = self.get_kl(self.actor, rollouts)    
            grads = torch.autograd.grad(d_kl, self.actor.parameters(), create_graph=True)
            grads = parameters_to_vector(grads)
            Jx = torch.sum(grads * x)
            Hx = torch.autograd.grad(Jx, self.actor.parameters())
            Hx = parameters_to_vector(Hx)
            return Hx + self.param.CG_DAMPING * x

        stepdir = self.conjugate_gradient(Hx, pg, self.param.NUM_CG_ITER)  
        stepsize = (2 * self.param.DELTA) / torch.dot(stepdir,Hx(stepdir))
        return torch.sqrt(stepsize) * stepdir

    def gae(self, rollouts):
        '''  Generaized Advantage Estimation '''
        states = torch.cat((rollouts.state,rollouts.next_state[-1:]))
        with torch.no_grad():
            values = self.critic(states).numpy()
        rewards = rollouts.reward.numpy()
        deltas = rewards + self.param.GAMMA * values[1:] - values[:-1]
        # rrlab magic discounting
        returns = scipy.signal.lfilter([1], [1, float(-self.param.GAMMA)], rewards[::-1], axis=0).astype('float32')
        advantages = scipy.signal.lfilter([1], [1, float(-self.param.GAMMA*self.param.LAMBDA)], deltas[::-1], axis=0).astype('float32')
        return torch.flip(torch.from_numpy(advantages),dims=[0]), torch.flip(torch.from_numpy(returns),dims=[0])

    def get_kl(self, model, rollouts):
        ''' Computes the KL-Divergance between the current policy and the model passed '''
        with torch.no_grad():
            p_old = self.actor.policy(rollouts['states'])
        p_new = model.policy(rollouts['states'])
        d_kl = kl_divergence(p_old, p_new).sum(dim=-1, keepdim=True).mean()
        return d_kl

    def conjugate_gradient(self, A, b, n):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = torch.dot(r,r)
        for i in range(n):
            if callable(A):
                Ap = A(p)
            else:
                Ap = torch.matmul(A,p)
            alpha = rs / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_next = torch.dot(r, r)
            betta = rs_next / rs
            p = r + betta * p
            rs = rs_next
            if rs < 1e-10:
                break
        return x

    def linesearch(self, npg, pg, rollouts):
        params_curr = parameters_to_vector(self.actor.parameters())
        for k in range(self.param.NUM_BACKTRACK):
            params_new = params_curr + self.param.ALPHA**k * npg 
            model_new = deepcopy(self.actor)
            vector_to_parameters(params_new, model_new.parameters())
            param_diff = params_new - params_curr
            surr_loss = torch.dot(pg,param_diff)
            kl_div = self.get_kl(model_new, rollouts)
            if surr_loss >= 0 and kl_div <= self.param.DELTA:
                params_curr = params_new
                break
        return params_curr, (self.param.ALPHA**k * npg).norm()