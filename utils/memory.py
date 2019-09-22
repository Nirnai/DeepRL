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

        self.states = np.zeros((capacity, env.observation_space.shape[0]), dtype=np.float32)
        self.actions = np.zeros((capacity, env.action_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, env.observation_space.shape[0]), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.returns = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)

        self.idx = 0
        self.start_idx = 0
    
    def __len__(self):
        return self.idx
    
    def store(self, state, action, reward, next_state, done):
        assert self.idx < self.max_size
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx += 1
    
    def process_episode(self, critic, actor, maximum_entropy=False):
        episode = slice(self.start_idx, self.idx)
        states = np.concatenate([self.states[episode], self.next_states[self.idx-1:self.idx]], axis = 0)
        rewards = self.rewards[episode]
        with torch.no_grad():
            states = torch.from_numpy(states).to(self.device)
            values = critic(states).cpu().numpy()
            if maximum_entropy:
                # Max Entropy RL
                rewards += self.tau * actor.entropy(states[:-1]).squeeze().cpu().numpy()
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        self.advantages[episode] = lfilter([1], [1, float(-self.gamma * self.lamda)], deltas[::-1], axis=0)[::-1]
        self.returns[episode] = lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.start_idx = self.idx


    def replay(self):
        assert self.idx == self.max_size
        self.idx, self.start_idx = 0, 0
        return dict(
            states = torch.from_numpy(self.states).to(self.device),
            actions = torch.from_numpy(self.actions).to(self.device),
            returns = torch.from_numpy(self.returns).to(self.device),
            advantages = torch.from_numpy(self.advantages).to(self.device)
        )



class ReplayBuffer():
    def __init__(self, capacity, rng, env, device):
        self.rng = rng
        self.device = device
        self.max_size = int(capacity)

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


# class Memory():
#     def __init__(self, capacity, rng, env, device):
#         self._rng = rng
#         self._device = device
#         self._max_size = int(capacity)

#         self._transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'initial'))
#         self._transition_type = [('state', np.float64, env.observation_space.shape), 
#                                  ('action', np.float64, env.action_space.shape),
#                                  ('reward', np.float64),
#                                  ('next_state', np.float64, env.observation_space.shape),
#                                  ('mask', np.int64),
#                                  ('initial', np.int64)]
#         self._buffer = np.zeros(self._max_size, dtype = self._transition_type)
#         self._len = 0
#         self._curr_idx = 0
#         self._next_idx = 0
        
    
#     def __len__(self):
#         # return len(self._buffer)
#         return self._len

#     def __getitem__(self, index):
#         return self._buffer[index]
    
#     def clear(self): 
#         # self.buffer.clear()
#         self._buffer = np.zeros(self._max_size, dtype = self._transition_type)
#         self._len = 0
    
#     def push(self, state, action, reward, next_state, done, initial):
#         """Saves a transition."""
#         self._buffer[self._next_idx] = (state, action, np.array([reward]), next_state, 1-done, initial)
#         if self._len < self._max_size:
#             self._len += 1
#         self._curr_idx = self._next_idx
#         self._next_idx = (self._next_idx + 1) % self._max_size

#     def sample(self, batch_size):
#         idxs = np.random.randint(0, self._len, batch_size)
#         transitions = self._buffer[idxs]
#         state = torch.from_numpy(transitions['state']).float().to(self._device) 
#         action = torch.from_numpy(transitions['action']).float().to(self._device)
#         reward = torch.from_numpy(transitions['reward']).float().to(self._device)
#         next_state = torch.from_numpy(transitions['next_state']).float().to(self._device)
#         mask = torch.from_numpy(transitions['mask']).float().to(self._device)
        
#         initial =  torch.from_numpy(transitions['initial']).float().to(self._device)
        
#         return self._transition(state, action, reward, next_state, mask, initial)

#     def replay(self, n=None):
#         if n is None:
#             if(self._len < self._max_size):
#                 transitions = self._buffer[:self._next_idx]
#             else:
#                 transitions = self._buffer[:]
#         else:
#             transitions = self._buffer[self._next_idx-n : self._next_idx]
#         state = torch.from_numpy(transitions['state']).float().to(self._device) 
#         action = torch.from_numpy(transitions['action']).float().to(self._device)
#         reward = torch.from_numpy(transitions['reward']).float().to(self._device)
#         next_state = torch.from_numpy(transitions['next_state']).float().to(self._device)
#         mask = torch.from_numpy(transitions['mask']).float().to(self._device)

#         initial =  torch.from_numpy(transitions['initial']).float().to(self._device)

#         return self._transition(state, action, reward, next_state, mask, initial)


