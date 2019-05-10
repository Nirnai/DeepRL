import random 
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist


class A2C():
    def __init__(self, env, param):
        self.name = "A2C"
        self.env = env
        self.param = param
        self.rng = random.Random()

        if self.param.SEED != None:
            self.seed(self.param.SEED)

        self.ac = ActorCritic(self.param.ACTOR_ARCHITECTURE, self.param.ACTIVATION)
        # self.value = Value(self.param.CRITIC_ARCHITECTURE, self.param.ACTIVATION)
        # self.ac = ActorCritic(self.value, self.policy)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.param.LEARNING_RATE)

        self.rewards = []
        self.log_probs = []
        self.values = []
        # self.advantages = []
        self.done = False
    
    def act(self, state):
        s = torch.from_numpy(state).float()
        probs, value = self.ac(s)
        m = dist.Categorical(probs)
        action = m.sample()
        next_state, reward, self.done, _ = self.env.step(action.numpy()) 
        s_next = torch.from_numpy(next_state).float()

        # _, value_next = self.ac(s_next)
        # advantage = reward + value_next - value

        self.rewards.append(reward) 
        self.log_probs.append(m.log_prob(action))
        self.values.append(value)
        # self.advantages.append(advantage)

        return next_state, reward, self.done


    def learn(self):
        if self.done:
            V_hat = torch.stack(self.values).view(1,-1)[0]
            V_target = torch.Tensor(self.monte_carlo_estimate(self.rewards))
            criterion = nn.MSELoss()
            value_loss = [nn.functional.smooth_l1_loss(value, torch.Tensor([R])) for R, value in zip(V_target, self.values)]

            advantages = [R - value.item() for R, value in zip(V_target, self.values)]
            policy_loss = [-log_prob * advantage for log_prob, advantage in zip(self.log_probs, advantages)]

            loss = torch.stack(policy_loss).sum()+ torch.stack(value_loss).sum()
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            del self.rewards[:]
            del self.log_probs[:]
            del self.values[:]


    def monte_carlo_estimate(self, rewards):
        R = 0
        V = []
        for r in rewards[::-1]:
            R = r + self.param.GAMMA * R
            V.insert(0, R)   
        V = torch.Tensor(V)
        V = V - V.mean()
        return V


    def seed(self, seed):
        self.param.SEED = seed
        torch.manual_seed(self.param.SEED)
        numpy.random.seed(self.param.SEED)
        self.rng = random.Random(self.param.SEED)

    def reset(self):
        self.__init__(self.env, self.param)



    def n_step_estimate(self, rewards, dones):
        pass
        # R = numpy.array([])
        # rewards.reverse()
        # # Inference for nth step
        # if dones[-1] is True:
        #     V_next = 0
        # else:
        #     V_next = self.value(torch.from_numpy(self.state_batch[-1]).float()).detach()
        # R = numpy.append(R, V_next)

        # # Bootstrap for prev steps
        # for r in rewards[1:]:
        #     V = r + self.param.GAMMA * V_next
        #     R = numpy.append(R, V)
        #     V_next = V
        # return R[::-1] * numpy.invert(numpy.array(dones))





class ActorCritic(nn.Module):
    def __init__(self, architecture, activation):
        super(ActorCritic, self).__init__()
        activation = getattr(nn.modules.activation, activation)()
        layers = [self.activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        self.layers = nn.Sequential(*layers)
        self.p = self.output_layer_policy(architecture[-2], architecture[-1])
        self.v = self.output_layer_value(architecture[-2])

    def forward(self, state):
        x = state
        x = self.layers(x)
        p = self.p(x)
        v = self.v(x)
        return p, v


    def activated_layer(self, in_, out_, activation_):
        return nn.Sequential(
            nn.Linear(in_, out_),
            activation_
        )
    

    def output_layer_policy(self, in_, out_):
        return nn.Sequential(
            nn.Linear(in_, out_),
            nn.Softmax()
        )

    def output_layer_value(self, in_):
        return nn.Sequential(
            nn.Linear(in_, 1)
        )


class Value(nn.Module):
    def __init__(self, architecture, activation):
        super(Value, self).__init__()
        activation = getattr(nn.modules.activation, activation)()
        layers = [self.activated_layer(in_, out_, activation) for in_, out_ in zip(architecture[:-1], architecture[1:-1])]
        self.layers = nn.Sequential(*layers)
        self.output = self.output_layer(architecture[-2], architecture[-1])

    def forward(self, state):
        x = state
        x = self.layers(x)
        y = self.output(x)
        return y


    def activated_layer(self, in_, out_, activation_):
        return nn.Sequential(
            nn.Linear(in_, out_),
            activation_
        )
    

    def output_layer(self, in_, out_):
        return nn.Sequential(
            nn.Linear(in_, out_)
        )