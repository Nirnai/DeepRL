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
        self.policy = CrossEntropyGuidedPolicy( self.Q, self.param.policy, self.device)
        self.steps = 0


    def act(self, state, deterministic=False):
        ######################################
        initial = False
        # if self.env.last_u is None:
        #     initial = True
        if self.env.timestep.step_type == 0:
            initial = True
        ######################################
        with torch.no_grad():
            action = self.policy(torch.from_numpy(state).float().to(self.device)).cpu().numpy()
        if deterministic is False:
            if self.param.POLICY_EXPLORATION_NOISE != 0:
                    action += np.random.randn(*action.shape) * self.param.POLICY_EXPLORATION_NOISE 
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_state, reward, done, _ = self.env.step(action)
        self.memory.push(state, action, reward, next_state, done, initial) 
        self.steps += 1
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData 

        # Update Q-Function
        next_actions = self.policy(batch.next_state)
        with torch.no_grad():
            q1_next, q2_next = self.Q.target(batch.next_state, next_actions) 
            q_targets = batch.reward + self.param.GAMMA * torch.min(q1_next, q2_next)
        q1, q2 = self.Q(batch.state, batch.action)
        loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets) 
        self.Q.optimize(loss)

        metrics = dict()
        metrics['Value Loss'] = loss.item()
        return metrics