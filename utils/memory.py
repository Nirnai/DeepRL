import torch
import numpy as np
from collections import deque, namedtuple


class Memory():
    def __init__(self, capacity, rng):
        self._transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._buffer = []
        self._max_size = int(capacity)
        self._next_idx = 0
        self._rng = rng
    
    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]
    
    def clear(self): 
        self.buffer.clear()
    
    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        mask = 1 - done
        if self._next_idx >= len(self):
            # self._buffer = np.append(self._buffer, self._transition(state, action, reward, next_state, mask))
            self._buffer.append(self._transition(state, action, reward, next_state, mask))
        else:
            self._buffer[self._next_idx] = self._transition(state, action, reward, next_state, mask)
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        # data = self.rng.sample(self.buffer, batch_size)
        idxs = np.random.randint(0, len(self), batch_size)
        transitions = self._transition(*zip(*[self._buffer[idx] for idx in idxs]))
        state = torch.from_numpy(np.array(transitions.state)).to(self._device)
        action = torch.from_numpy(np.array(transitions.action)).to(self._device)
        # action = torch.stack(transitions.action)
        reward = torch.from_numpy(np.array(transitions.reward)).to(self._device)
        next_state = torch.from_numpy(np.array(transitions.next_state)).to(self._device)
        mask = torch.from_numpy(np.array(transitions.mask)).to(self._device)
        return self._transition(state, action, reward, next_state, mask)

    def replay(self):
        transitions = self._transition(*zip(*self._buffer))
        state = torch.Tensor(transitions.state).to(self._device, non_blocking=True)
        action = torch.stack(transitions.action).to(self._device, non_blocking=True)
        reward = torch.Tensor(transitions.reward).to(self._device, non_blocking=True)
        next_state = torch.Tensor(transitions.next_state).to(self._device, non_blocking=True)
        mask = torch.Tensor(transitions.mask).to(self._device, non_blocking=True)
        return self._transition(state, action, reward, next_state, mask)