import torch
import numpy as np
from collections import deque, namedtuple


class Memory():
    def __init__(self, capacity, rng):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.buffer = deque(maxlen=int(capacity))
        self.rng = rng
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]
    
    def clear(self): 
        self.buffer.clear()
    
    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        mask = 1 - done
        self.buffer.append(self.transition(state, action, reward, next_state, mask))

    def sample(self, batch_size):
        transitions = self.transition(*zip(*self.rng.sample(self.buffer, batch_size)))
        state = torch.Tensor(transitions.state)
        action = torch.stack(transitions.action)
        reward = torch.Tensor(transitions.reward)
        next_state = torch.Tensor(transitions.next_state)
        mask = torch.Tensor(transitions.mask)
        return self.transition(state, action, reward, next_state, mask)
    
    def replay(self):
        transitions = self.transition(*zip(*self.buffer))
        state = torch.Tensor(transitions.state)
        action = torch.stack(transitions.action)
        reward = torch.Tensor(transitions.reward)
        next_state = torch.Tensor(transitions.next_state)
        mask = torch.Tensor(transitions.mask)
        return self.transition(state, action, reward, next_state, mask)