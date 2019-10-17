import torch
import numpy as np
from scipy.signal import lfilter
from collections import deque, namedtuple



class Buffer():
    def __init__(self, capacity, gamma, lamda, tau, env, device):
        self.device = device
        self.max_size = capacity
        self.gamma = gamma
        self.lamda = lamda
        self.tau = tau

        # Transition Data   
        self.states = np.zeros((capacity, env.observation_space.shape[0]), dtype=np.float32)
        self.actions = np.zeros((capacity, env.action_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, env.observation_space.shape[0]), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Predictor Data
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)

        # On-policy specific
        self.returns_mc = np.zeros(capacity, dtype=np.float32)
        self.returns_gae = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)

        self.idx = 0
        self.start_idx = 0
    
    def __len__(self):
        return self.idx
    
    def store(self, state, action, reward, next_state, done, value, next_value, log_pi):
        assert self.idx < self.max_size
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.values[self.idx] = value
        self.next_values[self.idx] = next_value
        self.log_probs[self.idx] = log_pi
        self.idx += 1

    
    def process_episode(self, maximum_entropy=False, mc_returns=False):
        episode = slice(self.start_idx, self.idx)
        rewards = self.rewards[episode]
        values = self.values[episode]
        next_values = self.next_values[episode]
        log_probs = self.log_probs[episode]
        dones = self.dones[episode]
        if maximum_entropy:
            # Max Entropy RL
            # rewards += self.tau * actor.entropy(states[:-1]).squeeze().cpu().numpy()
            rewards -= self.tau * log_probs
        deltas = rewards + self.gamma * next_values - values
        self.advantages[episode] = lfilter([1], [1, float(-self.gamma * self.lamda)], deltas[::-1], axis=0)[::-1]
        self.returns_mc[episode] = lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1] 
        self.returns_gae[episode] = self.advantages[episode] + values
        self.start_idx = self.idx



    def replay(self):
        assert self.idx == self.max_size
        if(self.start_idx < self.idx):
            self.process_episode()
        self.idx, self.start_idx = 0, 0
        return dict(
            states = torch.from_numpy(self.states).to(self.device),
            actions = torch.from_numpy(self.actions).to(self.device),
            returns_mc = torch.from_numpy(self.returns_mc).to(self.device),
            returns_gae = torch.from_numpy(self.returns_mc).to(self.device),
            advantages = torch.from_numpy(self.advantages).to(self.device),
            values = torch.from_numpy(self.values).to(self.device),
            log_probs = torch.from_numpy(self.log_probs).to(self.device)
        )


class ReplayBuffer():
    def __init__(self, capacity, rng, env, device):
        self.rng = rng
        self.device = device
        self.max_size = int(capacity)

        self.states_mean = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        self.states_std = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        self.rewards_mean = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        self.rewards_std = np.zeros(env.observation_space.shape[0], dtype=np.float32)

        self.states      = np.zeros((int(capacity), env.observation_space.shape[0]), dtype=np.float32)
        self.actions     = np.zeros((int(capacity), env.action_space.shape[0]), dtype=np.float32)
        self.rewards     = np.zeros(int(capacity), dtype=np.float32)
        self.next_states = np.zeros((int(capacity), env.observation_space.shape[0]), dtype=np.float32)
        self.dones       = np.zeros(int(capacity), dtype=np.float32)

        self.idx  = 0
        self.size = 0

    def __len__(self):
        return self.size

    def store(self, state, action, reward, next_state, done):
        self.states[self.idx]      = state
        self.actions[self.idx]     = action
        self.rewards[self.idx]     = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx]       = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0,self.size, size=batch_size)
        return dict(
            states = torch.from_numpy(self.states[idxs]).to(self.device),
            actions = torch.from_numpy(self.actions[idxs]).to(self.device),
            rewards = torch.from_numpy(self.rewards[idxs]).to(self.device),
            next_states = torch.from_numpy(self.next_states[idxs]).to(self.device),
            dones = torch.from_numpy(self.dones[idxs]).to(self.device)
        )

