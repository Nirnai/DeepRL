import gym
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display
from dm_control import suite
import pyglet
from pyglet.gl import *

class DeepMindControlSuiteWrapper(gym.core.Env):
    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward)        
        self.observation_spec = self.env.observation_spec()
        self.observation_dim = sum([np.int(np.prod(self.observation_spec[key].shape)) for key in self.observation_spec])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.observation_dim,))
        self.action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box( self.action_spec.minimum,  self.action_spec.maximum)
        self._max_episode_steps = self.env._step_limit

        self.timestep = None

        self.fig = None
        self.ax = None
        self.img = None
        self.background = None

    def step(self, action):
        self.timestep = self.env.step(action)
        next_state = np.concatenate([np.expand_dims(ob,axis=0) if ob.ndim is 0 else ob for ob in list(self.timestep.observation.values())])
        reward = self.timestep.reward
        done = self.timestep.last()
        return next_state, reward, done, {}


    def reset(self):
        self.timestep = self.env.reset()
        return np.concatenate([np.expand_dims(ob,axis=0) if ob.ndim is 0 else ob for ob in list(self.timestep.observation.values())])



    def render(self, mode='human',**kwargs):
        kwargs['camera_id'] = 0
        pixels = self.env.physics.render(**kwargs)
        if mode is 'human':
            plt.ion()
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.fig.canvas.draw()
                self.img = self.ax.imshow(pixels)
                self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            else:
                self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self.img.set_data(pixels)
                self.fig.canvas.restore_region(self.background)
                self.ax.draw_artist(self.img)
                self.fig.canvas.blit(self.ax.bbox)
        return pixels

