import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from algorithms import OffPolicy, BaseRL, ActionValueFunction
from utils.policies import CrossEntropyGuidedPolicy

class CGP(BaseRL, OffPolicy):
    def __init__(self, env):
        super(CGP, self).__init__(env, device="cpu")        
        self.name = "CGP"
        self.Q = ActionValueFunction(self.param.qvalue, self.device)
        self.policy = CrossEntropyGuidedPolicy(self.Q, self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        # ######################################
        # initial = False
        # # if self.env.last_u is None:
        # #     initial = True
        # if self.env.timestep.step_type == 0:
        #     initial = True
        # ######################################
        with torch.no_grad():
            action = self.policy(torch.from_numpy(state).float().to(self.device))
        if deterministic is False and self.param.POLICY_EXPLORATION_NOISE != 0:
            action += torch.randn(action.shape).to(self.device) * self.param.POLICY_EXPLORATION_NOISE 
            action = action.cpu().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        else:
            action = action.cpu().numpy()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_state, reward, done, _ = self.env.step(action)
        self.memory.store(state, action, reward, next_state, done) 
        self.steps += 1
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData 

        # # Add noise on experience
        # noise = torch.randn(batch['actions'].shape).to(self.device) * self.param.POLICY_UPDATE_NOISE
        # noise = torch.clamp(noise, -self.param.POLICY_CLIP, self.param.POLICY_CLIP)
        # next_actions = self.policy(batch['next_states']) + noise

        # Update Q-Function
        next_actions = self.policy(batch['next_states'])
        with torch.no_grad():
            # q1_next, q2_next = self.Q.target(batch['next_states'], next_actions) 
            q_next = self.Q.target(batch['next_states'], next_actions)
            q_targets = batch['rewards'] + self.param.GAMMA * q_next#torch.min(q1_next, q2_next)
        # q1, q2 = self.Q(batch['states'], batch['actions'])
        q = self.Q(batch['states'], batch['actions'])
        # loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets) 
        loss = F.mse_loss(q, q_targets)
        self.Q.optimize(loss)

        # Return Metrics
        metrics = dict()
        if self.steps % 5000 == 0:
            data = self.memory.sample(5000)
            states = data['states']
            actions = self.policy(states)
            q = self.Q(states, actions)
            metrics['value'] = q.mean().item()
        return metrics