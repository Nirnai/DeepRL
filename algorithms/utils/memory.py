import numpy as np
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity, rng):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append(Transition(state, action, reward, next_state, done))      

    def sample(self, batch_size):
        transitions = Transition(*zip(*self.rng.sample(self.buffer, batch_size)))
        state_batch = np.concatenate(transitions.state)
        action_batch = np.array(transitions.action)
        reward_batch = np.array(transitions.reward)
        next_state_batch = np.concatenate(transitions.next_state)
        done_batch = np.float32(transitions.done)
        transitions = Transition(state_batch, 
                                 action_batch, 
                                 reward_batch, 
                                 next_state_batch, 
                                 done_batch)
        return transitions
        
    def __len__(self):
        return len(self.buffer)