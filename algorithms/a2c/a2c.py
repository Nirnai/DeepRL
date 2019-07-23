import torch
from algorithms import onPolicy, ActorCritic, HyperParameter

class A2C(ActorCritic):
    def __init__(self, env):
        super().__init__(env)   
        self.name = "A2C"

    @onPolicy
    def learn(self):
        rollouts = self.onPolicyData
        # Generalized Advantage Estimation
        advantages = self.returns(rollouts) - self.values(rollouts)
        # Critic Step
        critic_loss = advantages.pow(2).mean()
        self.optimize_critic(critic_loss)
        # Actor Step
        actor_loss = self.pg_objective(rollouts, advantages)
        self.optimize_actor(actor_loss)

        return critic_loss, self.actor.entropy(rollouts.state).sum().item()


    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def pg_objective(self, rollouts, advantages):
        entropy = self.actor.policy(rollouts.state).entropy().squeeze()
        log_probs = self.actor.log_probs(rollouts.state, rollouts.action)
        loss = (- log_probs * advantages.detach()).sum()
        return loss

    def returns(self, rollouts):
        ''' Computes Monte Carlo Returns '''
        returns = [0] * len(rollouts.reward)
        for t in reversed(range(len(rollouts.reward[:-1]))):
            returns[t] = rollouts.reward[t] + self.param.GAMMA * rollouts.mask[t] * returns[t+1]
        returns = torch.Tensor(returns)
        return returns

    def gae(self, rollouts):
        '''  Generalized Advantage Estimation '''
        values = self.critic(rollouts.state).squeeze()
        with torch.no_grad():
            next_value = self.critic(rollouts.next_state[-1])
        values = torch.cat((values, next_value))
        advantages = [0] * (len(rollouts.reward)+1)
        for t in reversed(range(len(rollouts.reward))):
            delta = rollouts.reward[t] + self.param.GAMMA * values[t+1] * rollouts.mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * rollouts.mask[t] * advantages[t+1]
        advantages = torch.stack(advantages[:-1])
        return advantages 
            
    def values(self, rollouts):
        return self.critic(rollouts.state).squeeze()