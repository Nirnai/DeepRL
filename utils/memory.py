import torch
import numpy as np
from collections import deque, namedtuple


class Memory():
    def __init__(self, capacity, rng, env, device):
        self._rng = rng
        self._device = device
        self._max_size = int(capacity)

        self._transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self._transition_type = [('state', np.float64, env.observation_space.shape), 
                                 ('action', np.float64, env.action_space.shape),
                                 ('reward', np.float64),
                                 ('next_state', np.float64, env.observation_space.shape),
                                 ('mask', np.int64)]
        self._buffer = np.zeros(self._max_size, dtype = self._transition_type)
        self._len = 0
        self._next_idx = 0
        
    
    def __len__(self):
        # return len(self._buffer)
        return self._len

    def __getitem__(self, index):
        return self._buffer[index]
    
    def clear(self): 
        # self.buffer.clear()
        self._buffer = np.zeros(self._max_size, dtype = self._transition_type)
        self._len = 0
    
    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        # mask = 1 - done
        # if self._next_idx >= len(self):
        #     self._buffer.append(self._transition(state, action, reward, next_state, mask))
        # else:
        #     self._buffer[self._next_idx] = self._transition(state, action, reward, next_state, mask)

        self._buffer[self._next_idx] = (state, action, np.array([reward]), next_state, 1-done)
        if self._len < self._max_size:
            self._len += 1
        
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self._len, batch_size)
        transitions = self._buffer[idxs]
        state = torch.from_numpy(transitions['state']).float().to(self._device) 
        action = torch.from_numpy(transitions['action']).float().to(self._device)
        reward = torch.from_numpy(transitions['reward']).float().to(self._device)
        next_state = torch.from_numpy(transitions['next_state']).float().to(self._device)
        mask = torch.from_numpy(transitions['mask']).float().to(self._device)
        return self._transition(state, action, reward, next_state, mask)

    def replay(self):
        transitions = self._buffer[:]
        state = torch.from_numpy(transitions['state']).float().to(self._device) 
        action = torch.from_numpy(transitions['action']).float().to(self._device)
        reward = torch.from_numpy(transitions['reward']).float().to(self._device)
        next_state = torch.from_numpy(transitions['next_state']).float().to(self._device)
        mask = torch.from_numpy(transitions['mask']).float().to(self._device)
        return self._transition(state, action, reward, next_state, mask)