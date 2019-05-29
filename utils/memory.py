import numpy as np
from collections import deque, namedtuple

class RolloutBuffer(object):
    def __init__(self):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.memory = []

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index):
        return self.memory[index]

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        mask = 1 - done
        self.memory.append(self.transition(state, action, reward, next_state, mask))

    def sample(self):
        return self.transition(*zip(*self.memory))


class ReplayBuffer(object):
    def __init__(self, capacity, rng):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.buffer = deque(maxlen=capacity)
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        mask = 1 - done
        self.buffer.append(self.transition(state, action, reward, next_state, mask))      

    def sample(self, batch_size):
        transitions = self.transition(*zip(*self.rng.sample(self.buffer, batch_size)))
        return transitions
        
    def __len__(self):
        return len(self.buffer)


    
if __name__ == '__main__':

    experience = RolloutBuffer()
    for i in range(10):
        experience.push(i,i,i,i,True)
        
    print(len(experience))
    print(experience[5])
