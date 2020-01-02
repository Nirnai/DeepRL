# DeepRL
DeepRL is a pytorch implementation of Deep Reinforcement Learning algorithms. The project contains the following modules:
- models
- algorithms
- evaluator

## Models
Models implements a set of common policies and value functions. 

Implemented policies are:
- Gaussian
- Gaussian Bounded
- Gaussian Clipped
- Deterministic
- Cross Entropy Method (CEM) 

Implemented value functions are:
- ValueFunction
- ActionValueFunction

## Algorithms
The algorithms module implements a base class (BaseRL) for RL agents. 
Implemented algorithms are:

Trust Region Policy Optimization (TRPO) 
  - Original Paper: https://arxiv.org/abs/1502.05477
  
Proximal Policy Optimization (PPO)
  - Original Paper: https://arxiv.org/abs/1707.06347
  
Twin Delayed Deep Deterministic Policy Gradient (TD3)
  - Original Paper: https://arxiv.org/abs/1802.09477
  
Soft Actor-Critic (SAC)
  - Original Paper: https://arxiv.org/abs/1801.01290
  
Cross Entropy Guided Policy (CGP)
  - Original Paper: https://arxiv.org/abs/1903.10605

An example of the exposed interface is shown in the following:
```python
import gym
from algorithms import TD3

env = gym.make('Pendulum-v1')
agent = TD3(env)
done = True
for t in range(timesteps):
  if done:
      state = env.reset()
      state, reward, done = agent.act(state)
      metrics = agent.learn()
```

## Evaluator
The evaluator module addresses the issue of reproducable RL, by seperating the evaluation process from the core algorithms.
The evaluation class allows to easily generate statistical evaluation of an algorithm

```python
import gym 
import dm_control2gym
from algorithms import PPO, TRPO, SAC, CGP, TD3
from evaluator import Evaluator
from evaluator.plot import plot_learning_curves, load_dataset

envs = [
    ('cartpole', 'balance'),
    ('cartpole', 'swingup'),
    ('acrobot', 'swingup'),
    ('cheetah', 'run'),
    ('hopper', 'hop'),
    ('walker', 'run')
]

algs = [TD3, CGP, SAC, TRPO, PPO]

for alg in algs:
  for domain, task in envs:
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        agent = alg(env)
        evl = Evaluator(agent, 'path/to/output/data')
        evl.run_statistic(samples=20, seed=0)
returns = load_dataset('path/to/returns_offline.npz')
fig, ax = plot_learning_curves(returns, interval='bs')
```


## Requirements
- Python3
- PyTorch
- Gym
- MuJoCo
- dm_control
- dm_control2gym
