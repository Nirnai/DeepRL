import torch
import numpy as np
from collections import deque, namedtuple


class Memory():
    def __init__(self, capacity, rng):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # state = torch.from_numpy(np.array(list(np.array([*self.rng.sample(self.buffer, batch_size)])[:,0]), dtype=np.float))
        transitions = self.transition(*zip(*self.rng.sample(self.buffer, batch_size)))
        state = torch.FloatTensor(transitions.state).to(self.device, non_blocking=True)
        action = torch.stack(transitions.action).to(self.device, non_blocking=True)
        reward = torch.FloatTensor(transitions.reward).to(self.device, non_blocking=True)
        next_state = torch.FloatTensor(transitions.next_state).to(self.device, non_blocking=True)
        mask = torch.Tensor(transitions.mask).to(self.device, non_blocking=True)
        return self.transition(state, action, reward, next_state, mask)
    
    def replay(self):
        transitions = self.transition(*zip(*self.buffer))
        state = torch.Tensor(transitions.state).to(self.device, non_blocking=True)
        action = torch.stack(transitions.action).to(self.device, non_blocking=True)
        reward = torch.Tensor(transitions.reward).to(self.device, non_blocking=True)
        next_state = torch.Tensor(transitions.next_state).to(self.device, non_blocking=True)
        mask = torch.Tensor(transitions.mask).to(self.device, non_blocking=True)
        return self.transition(state, action, reward, next_state, mask)