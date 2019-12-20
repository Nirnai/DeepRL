# DeepRL
This module contains pytorch implementations of several deep reinforcement learning algorithms. 

## Requirements
- Python3
- PyTorch
- Gym

## Training
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

## Evaluation
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
