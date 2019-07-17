import torch
import torch.distributions as dist
import torch.nn.functional as F
from algorithms import OffPolicy, BaseRL, QModel
from utils.policies import CrossEntropyGuidedPolicy


class CGP(BaseRL, OffPolicy):
    def __init__(self, env):
        super(CGP, self).__init__(env)
        self.name = "CGP"
        self.Q = QModel(self.param)
        self.policy = CrossEntropyGuidedPolicy( self.Q, 
                                                self.param.CEM_ITERATIONS, 
                                                self.param.CEM_BATCH, 
                                                self.param.CEM_TOPK)
        self.steps = 0


    def act(self, state):
        action = self.policy(torch.from_numpy(state).float())
        next_state, reward, done, _ = self.env.step(action.numpy())
        self._memory.push(state, action, reward, next_state, done) 
        self.steps += 1
        if done:
            next_state = self.env.reset()
        return next_state, reward, done


    @OffPolicy.loop
    def learn(self):
        batch = self.offPolicyData 

        # Update Q-Function
        next_actions = self.policy(batch.next_state)
        with torch.no_grad():
            q1_next, q2_next = self.Q.target(batch.next_state, next_actions) 
            # TODO: Mask or no Mask?
            q_targets = batch.reward + self.param.GAMMA * torch.min(q1_next, q2_next)
        q1, q2 = self.Q(batch.state, batch.action)
        loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets) 
        self.Q.optimize(loss)

        metrics = dict()
        metrics['Value Loss'] = loss.item()
        return metrics



# class CGP(QLearning):
#     def __init__(self, env):
#         super(CGP, self).__init__(env)
#         self.name = "CGP"

#     def act(self, state, exploit=False):
#         if not exploit:
#             action = self.cem(torch.from_numpy(state).float(), self.Q1)
#             next_state, reward, done, _ = self.env.step(action.numpy())
#             self.memory.push(state, action, reward, next_state, done) 
#             self.steps += 1
#         if done:
#             next_state = self.env.reset()
#         return next_state, reward, done

#     @offPolicy
#     def learn(self):
#         batch = self.offPolicyData
        
#         # TODO: Double or not 
          # BUG!!!
#         next_actions = self.cem(batch.state, self.Q1)
        
#         with torch.no_grad():
#             Q_next = torch.min(self.Q1_target(batch.next_state, next_actions), self.Q2_target(batch.next_state, next_actions)).squeeze()
#             # TODO: Mask or no Mask?
#             Q_targets = batch.reward + self.param.GAMMA * Q_next
        
#         Q1_current = self.Q1(batch.state, batch.action).squeeze()
#         Q2_current = self.Q2(batch.state, batch.action).squeeze()

#         loss = F.mse_loss(Q1_current, Q_targets) + F.mse_loss(Q2_current, Q_targets) 
        
#         self.optimize_Q(loss)
#         self.soft_target_update()

#         metrics = dict()
#         metrics['Value Loss'] = loss.item()
#         return metrics



#     def cem(self, state, Q):
#         if state.dim() == 2:
#             mean = torch.zeros(state.shape[0],1)
#             std = torch.ones(state.shape[0],1)
#         else:
#             mean = torch.Tensor([0.0])
#             std = torch.Tensor([1.0])
        
#         for i in range(self.param.CEM_ITERATIONS):
#             p = dist.Normal(mean, std)
#             states = torch.cat(self.param.CEM_BATCH*[state.unsqueeze(0)], dim=0)
#             actions = p.sample((self.param.CEM_BATCH,))
#             with torch.no_grad():
#                 Qs = Q(states, actions)
#             Is = Qs.topk(self.param.CEM_TOPK , dim=0)[1].unsqueeze(-1)
#             mean = actions.gather(0, Is).mean(dim = 0)
#             std = actions.gather(0, Is).std(dim = 0)
#         return actions.gather(0, Is[0:1])[0]