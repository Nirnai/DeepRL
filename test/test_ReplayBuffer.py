import pytest
import numpy
import random
from algorithms.dqn.dqn import ReplayBuffer

def test_replay():

    rng = random.Random()
    replayBuffer = ReplayBuffer(10, rng)

    for i in range(10):
        replayBuffer.push(1,2,3,4,True)
    transitions = replayBuffer.sample(4)
    assert type(transitions.state) is numpy.ndarray and len(transitions.state) is 4
    assert type(transitions.action) is numpy.ndarray and len(transitions.action) is 4
    assert type(transitions.reward) is numpy.ndarray and len(transitions.reward) is 4
    assert type(transitions.next_state) is numpy.ndarray and len(transitions.next_state) is 4
    assert type(transitions.done) is numpy.ndarray and len(transitions.done) is 4