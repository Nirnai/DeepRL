import torch
from copy import deepcopy
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from algorithms import ActorCritic, HyperParameter, onPolicy


class TRPO(ActorCritic):
    def __init__(self, env):
        super(TRPO, self).__init__(env)
        self.name = "TRPO"

    @onPolicy
    def learn(self):
        rollouts = self.onPolicyData
        # Compute Advantages
        advantages = self.gae(rollouts)    
        # Critic Step
        critic_loss = advantages.pow(2.).mean()
        self.optimize_critic(critic_loss)
        # Actor Step
        pg = self.policy_gradient(advantages, rollouts)
        npg = self.natural_gradient(pg, rollouts)
        parameters = self.linesearch(npg, pg, rollouts)
        self.optimize_actor(parameters)

        metrics = dict()
        metrics['value loss'] = critic_loss.item()
        metrics['policy entropy'] = self.actor.entropy(rollouts.state).sum().item()
        return metrics
        

    ################################################################
    ########################## Utilities ###########################
    ################################################################
    def optimize_actor(self, new_parameters):
        vector_to_parameters(new_parameters, self.actor.parameters())

    def policy_gradient(self, advantages, rollouts):
        log_probs = self.actor.log_probs(rollouts.state, rollouts.action)
        pg_objective = (log_probs * advantages.detach()).mean()
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
        values = self.critic(rollouts.state).squeeze()
        with torch.no_grad():
            next_value = self.critic(rollouts.next_state[-1])
        values = torch.cat((values, next_value))
        advantages = [0] * (len(rollouts.reward) + 1 )
        for t in reversed(range(len(rollouts.reward))):
            delta = rollouts.reward[t] + self.param.GAMMA * values[t+1] * rollouts.mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * rollouts.mask[t] * advantages[t+1]
        advantages = torch.stack(advantages[:-1])
        return advantages#(advantages - advantages.mean()) / advantages.std() 

    def get_kl(self, model, rollouts):
        ''' Computes the KL-Divergance between the current policy and the model passed '''
        with torch.no_grad():
            p_old = self.actor.policy(rollouts.state)
        p_new = model.policy(rollouts.state)
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
        params_curr = self.actor.get_params()
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