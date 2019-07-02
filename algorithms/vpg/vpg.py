import torch
from algorithms import onPolicy, ActorOnly

class VPG(ActorOnly):
    def __init__(self, env):    
        super().__init__(env)    
        self.name = "VPG"
    
    @onPolicy
    def learn(self):
        ''' Vanilla Policy Gradient Algorithm '''
        rollouts = self.onPolicyData
        # Monte Carlo estimate of returns
        returns = self.mc_returns(rollouts)
        # Get Loss
        loss = self.pg_objective(returns, rollouts)
        # Take Gradient Step
        self.optimize_actor(loss)


    ################################################################
    ########################## Utilities ###########################
    ################################################################
    def pg_objective(self, returns, rollouts):
        # Log Probability of current policy
        log_probs = self.policy.log_probs(rollouts.state, rollouts.action)
        # Compute Loss L = -log(π(a,s)) * A
        loss = (- log_probs * returns).sum()
        return loss

    def mc_returns(self, rollouts):
        ''' Computes Monte Carlo Returns '''
        returns = [0] * len(rollouts.reward)
        for t in reversed(range(len(rollouts.reward[:-1]))):
            returns[t] = rollouts.reward[t] + self.param.GAMMA * rollouts.mask[t] * returns[t+1]
        returns = torch.Tensor(returns)
        return returns
    