import torch
import torch.nn.functional as F
import scipy.signal
from algorithms import BaseRL, OnPolicy, ValueFunction
from utils.policies import GaussianPolicy
from copy import deepcopy


class PPO(BaseRL, OnPolicy):
    def __init__(self, env):
        super(PPO, self).__init__(env)
        self.name = 'PPO'
        self.critic = ValueFunction(self.param.value , self.device)
        self.actor = GaussianPolicy(self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state).float().to(self.device), deterministic=deterministic).cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)
        if not deterministic:
            self.memory.store(state, action, reward, next_state, done)
            if done:
                self.memory.process_episode(self.critic, self.actor, maximum_entropy=False) 
        self.steps += 1
        return next_state, reward, done
    
    @OnPolicy.loop
    def learn(self):
        rollouts = self.onPolicyData
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        self.actor_old = deepcopy(self.actor)
        for _ in range(self.param.EPOCHS):
            for _ in range(self.param.VALUE_EPOCHS):
                # Compute Advantages
                values = self.critic(rollouts['states'])
                # Critic Step
                critic_loss = F.mse_loss(values, returns)
                self.critic.optimize(critic_loss)
            for i in range(self.param.POLICY_EPOCHS):
                # Actor Step
                # advantages = (advantages - advantages.mean()) / advantages.std() 
                actor_loss, kl = self.clipped_objective(rollouts, advantages) 
                if kl > 1.5 * 0.01:
                    print("Early stopping at step {} due to reaching max kl.".format(i))
                    break
                # # Entropy Regularization
                # entropy = self.actor.entropy(rollouts['states']).mean()
                # actor_loss += 0.01 * entropy
                self.actor.optimize(actor_loss)

        metrics = dict()
        metrics['value'] = values.mean().item()
        metrics['target'] = returns.mean().item()
        return metrics

    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def clipped_objective(self, rollouts, advantages):
        ratio, kl= self.importance_weights(rollouts['states'], rollouts['actions'])
        ratio_clip = torch.clamp(ratio, 1 - self.param.CLIP,  1 + self.param.CLIP)
        policy_loss = -torch.min(ratio * advantages, ratio_clip * advantages)
        return policy_loss.mean(), kl
        
    def importance_weights(self, states, actions): 
        with torch.no_grad():
            old_log_probs = self.actor_old.log_prob(states, actions)
        curr_log_probs = self.actor.log_prob(states, actions)
        kl = (old_log_probs-curr_log_probs).mean()
        ratio = torch.exp(curr_log_probs - old_log_probs)
        return ratio, kl