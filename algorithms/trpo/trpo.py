import torch
import torch.nn.functional as F
import scipy.signal
from copy import deepcopy
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from algorithms import BaseRL, OnPolicy, ValueFunction
from utils.policies import GaussianPolicy, BoundedGaussianPolicy


class TRPO(BaseRL, OnPolicy):
    def __init__(self, env):
        super(TRPO,self).__init__(env)
        self.name = "TRPO"
        self.critic = ValueFunction(self.param.value, self.device)
        self.actor = GaussianPolicy(self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.store(state, action, reward, next_state, done) 
        if done:
            self.memory.process_episode(self.critic, self.actor) 
        self.steps += 1
        return next_state, reward, done
    

    @OnPolicy.loop
    def learn(self):
        rollouts = self.onPolicyData
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        for _ in range(self.param.EPOCHS):
            # Compute Advantages
            for _ in range(self.param.VALUE_EPOCHS):
                # Update Critic
                values = self.critic(rollouts['states'])
                critic_loss = F.mse_loss(values, returns)
                self.critic.optimize(critic_loss)
            # Update Actor
            advantages = (advantages - advantages.mean()) / advantages.std()
            pg = self.policy_gradient(advantages, rollouts)
            npg = self.natural_gradient(pg, rollouts)
            parameters = self.linesearch(npg, pg, rollouts)
            self.optimize_actor(parameters)

        metrics = dict()
        metrics['value'] = values.mean().item()
        return metrics

    ################################################################
    ########################## Utilities ###########################
    ################################################################
    def optimize_actor(self, new_parameters):
        vector_to_parameters(new_parameters, self.actor.parameters())

    def policy_gradient(self, advantages, rollouts):
        log_probs = self.actor.log_prob(rollouts['states'], rollouts['actions'])
        pg_objective = (log_probs * advantages).mean()
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
        return params_curr