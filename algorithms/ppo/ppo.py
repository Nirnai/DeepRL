import torch
from algorithms import BaseRL, OnPolicy, VModel
from utils.policies import GaussianPolicy
from copy import deepcopy

class PPO(BaseRL, OnPolicy):
    def __init__(self, env):
        super(PPO, self).__init__(env)
        self.name = 'PPO'

        self.critic = VModel(self.param)
        self.actor = GaussianPolicy( self.param.ARCHITECTURE,
                                     self.param.ACTIVATION,
                                     self.param.ACTOR_LEARNING_RATE)
        self.steps = 0

    def act(self, state, deterministic=False):
        action = self.actor(torch.from_numpy(state).float())
        next_state, reward, done, _ = self.env.step(action.numpy())
        self._memory.push(state, action, reward, next_state, done) 
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done
    
    @OnPolicy.loop
    def learn(self):
        rollouts = self.onPolicyData
        self.actor_old = deepcopy(self.actor)
        for _ in range(self.param.EPOCHS):
            # Compute Advantages
            advantages = self.gae(rollouts)
            # Critic Step
            critic_loss = advantages.pow(2).mean()
            self.critic.optimize(critic_loss)
            # Actor Step
            actor_loss = self.clipped_objective(rollouts, advantages)
            self.actor.optimize(actor_loss)

        metrics = dict()
        metrics['value loss'] = critic_loss.item()
        metrics['policy entropy'] = self.actor.entropy(rollouts.state).sum().item()
        return metrics

# class PPO(ActorCritic):
#     def __init__(self, env):
#         super(PPO, self).__init__(env)
#         self.name = "PPO"
        

#     @onPolicy
#     def learn(self):
#         rollouts = self.onPolicyData
#         self.actor_old = deepcopy(self.actor)
#         for _ in range(self.param.EPOCHS):
#             # Compute Advantages
#             advantages = self.gae(rollouts)
#             # Critic Step
#             critic_loss = advantages.pow(2).mean()
#             self.optimize_critic(critic_loss)
#             # Actor Step
#             actor_loss = self.clipped_objective(rollouts, advantages)
#             self.optimize_actor(actor_loss)

#         metrics = dict()
#         metrics['value loss'] = critic_loss.item()
#         metrics['policy entropy'] = self.actor.entropy(rollouts.state).sum().item()
#         return metrics


    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def clipped_objective(self, rollouts, advantages):
        ratio = self.importance_weights(rollouts.state, rollouts.action)
        ratio_clip = torch.clamp(ratio, 1 - self.param.CLIP,  1 + self.param.CLIP)
        policy_loss = -torch.min(ratio * advantages.detach(), ratio_clip * advantages.detach()).mean()
        return policy_loss
        
    def importance_weights(self, states, actions): 
        with torch.no_grad():
            old_log_probs = self.actor_old.log_prob(states, actions)
        curr_log_probs = self.actor.log_prob(states, actions)
        ratio = torch.exp(curr_log_probs - old_log_probs)
        return ratio

    def gae(self, rollouts):
        '''  Generaized Advantage Estimation '''
        values = self.critic(rollouts.state).squeeze()
        with torch.no_grad():
            next_value = self.critic(rollouts.next_state[-1]).unsqueeze(-1)
        values = torch.cat((values, next_value))
        advantages = [0] * (len(rollouts.reward)+1)
        for t in reversed(range(len(rollouts.reward))):
            delta = rollouts.reward[t] + self.param.GAMMA * values[t+1] * rollouts.mask[t] - values[t]
            advantages[t] = delta + self.param.GAMMA * self.param.LAMBDA * rollouts.mask[t] * advantages[t+1]
        advantages = torch.stack(advantages[:-1])
        return (advantages - advantages.mean()) / advantages.std() 