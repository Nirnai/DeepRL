from abc import ABC, abstractmethod
import numpy as np

def getEnvInfo(env):
    state_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        action_space = 'discrete'
    else:
        action_dim = env.action_space.shape[0]
        action_space = 'continuous'

    return state_dim, action_dim, action_space


if __name__ == '__main__':
    import gym

    env = gym.make('Pendulum-v0')
    state_dim, action_dim, action_space = getEnvInfo(env)
    print('State Dimension: {}, Action Dimension: {}, Action Space: {}'.format(state_dim, action_dim, action_space))

# """
# Helpers for dealing with vectorized environments.
# """

# from collections import OrderedDict

# import gym
# import numpy as np


# def copy_obs_dict(obs):
#     """
#     Deep-copy an observation dict.
#     """
#     return {k: np.copy(v) for k, v in obs.items()}


# def dict_to_obs(obs_dict):
#     """
#     Convert an observation dict into a raw array if the
#     original observation space was not a Dict space.
#     """
#     if set(obs_dict.keys()) == {None}:
#         return obs_dict[None]
#     return obs_dict


# def obs_space_info(obs_space):
#     """
#     Get dict-structured information about a gym.Space.
#     Returns:
#       A tuple (keys, shapes, dtypes):
#         keys: a list of dict keys.
#         shapes: a dict mapping keys to shapes.
#         dtypes: a dict mapping keys to dtypes.
#     """
#     if isinstance(obs_space, gym.spaces.Dict):
#         assert isinstance(obs_space.spaces, OrderedDict)
#         subspaces = obs_space.spaces
#     else:
#         subspaces = {None: obs_space}
#     keys = []
#     shapes = {}
#     dtypes = {}
#     for key, box in subspaces.items():
#         keys.append(key)
#         shapes[key] = box.shape
#         dtypes[key] = box.dtype
#     return keys, shapes, dtypes


# def obs_to_dict(obs):
#     """
#     Convert an observation into a dict.
#     """
#     if isinstance(obs, dict):
#         return obs
#     return {None: obs}



# class VecEnv(ABC):
#     """
#     An abstract asynchronous, vectorized environment.
#     Used to batch data from multiple copies of an environment, so that
#     each observation becomes an batch of observations, and expected action is a batch of actions to
#     be applied per-environment.
#     """
#     closed = False
#     viewer = None

#     metadata = {
#         'render.modes': ['human', 'rgb_array']
#     }

#     def __init__(self, num_envs, observation_space, action_space):
#         self.num_envs = num_envs
#         self.observation_space = observation_space
#         self.action_space = action_space

#     @abstractmethod
#     def reset(self):
#         """
#         Reset all the environments and return an array of
#         observations, or a dict of observation arrays.
#         If step_async is still doing work, that work will
#         be cancelled and step_wait() should not be called
#         until step_async() is invoked again.
#         """
#         pass

#     @abstractmethod
#     def step_async(self, actions):
#         """
#         Tell all the environments to start taking a step
#         with the given actions.
#         Call step_wait() to get the results of the step.
#         You should not call this if a step_async run is
#         already pending.
#         """
#         pass

#     @abstractmethod
#     def step_wait(self):
#         """
#         Wait for the step taken with step_async().
#         Returns (obs, rews, dones, infos):
#          - obs: an array of observations, or a dict of
#                 arrays of observations.
#          - rews: an array of rewards
#          - dones: an array of "episode done" booleans
#          - infos: a sequence of info objects
#         """
#         pass

#     def close_extras(self):
#         """
#         Clean up the  extra resources, beyond what's in this base class.
#         Only runs when not self.closed.
#         """
#         pass

#     def close(self):
#         if self.closed:
#             return
#         if self.viewer is not None:
#             self.viewer.close()
#         self.close_extras()
#         self.closed = True

#     def step(self, actions):
#         """
#         Step the environments synchronously.
#         This is available for backwards compatibility.
#         """
#         self.step_async(actions)
#         return self.step_wait()

#     def render(self, mode='human'):
#         imgs = self.get_images()
#         bigimg = tile_images(imgs)
#         if mode == 'human':
#             self.get_viewer().imshow(bigimg)
#             return self.get_viewer().isopen
#         elif mode == 'rgb_array':
#             return bigimg
#         else:
#             raise NotImplementedError

#     def get_images(self):
#         """
#         Return RGB images from each environment
#         """
#         raise NotImplementedError

#     @property
#     def unwrapped(self):
#         if isinstance(self, VecEnvWrapper):
#             return self.venv.unwrapped
#         else:
#             return self

#     def get_viewer(self):
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.SimpleImageViewer()
#         return self.viewer





# class DummyVecEnv(VecEnv):
#     """
#     VecEnv that does runs multiple environments sequentially, that is,
#     the step and reset commands are send to one environment at a time.
#     Useful when debugging and when num_env == 1 (in the latter case,
#     avoids communication overhead)
#     """
#     def __init__(self, env_fns):
#         """
#         Arguments:
#         env_fns: iterable of callables      functions that build environments
#         """
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
#         obs_space = env.observation_space
#         self.keys, shapes, dtypes = obs_space_info(obs_space)

#         self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
#         self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
#         self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
#         self.buf_infos = [{} for _ in range(self.num_envs)]
#         self.actions = None
#         self.spec = self.envs[0].spec

#     def step_async(self, actions):
#         listify = True
#         try:
#             if len(actions) == self.num_envs:
#                 listify = False
#         except TypeError:
#             pass

#         if not listify:
#             self.actions = actions
#         else:
#             assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
#             self.actions = [actions]

#     def step_wait(self):
#         for e in range(self.num_envs):
#             action = self.actions[e]
#             # if isinstance(self.envs[e].action_space, spaces.Discrete):
#             #    action = int(action)

#             obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
#             if self.buf_dones[e]:
#                 obs = self.envs[e].reset()
#             self._save_obs(e, obs)
#         return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
#                 self.buf_infos.copy())

#     def reset(self):
#         for e in range(self.num_envs):
#             obs = self.envs[e].reset()
#             self._save_obs(e, obs)
#         return self._obs_from_buf()

#     def _save_obs(self, e, obs):
#         for k in self.keys:
#             if k is None:
#                 self.buf_obs[k][e] = obs
#             else:
#                 self.buf_obs[k][e] = obs[k]

#     def _obs_from_buf(self):
#         return dict_to_obs(copy_obs_dict(self.buf_obs))

#     def get_images(self):
#         return [env.render(mode='rgb_array') for env in self.envs]

#     def render(self, mode='human'):
#         # if self.num_envs == 1:
#         return self.envs[0].render(mode=mode)
#         # else:
#         #     return super().render(mode=mode)





# if __name__ == '__main__':

#     env_name = 'Pendulum-v0'
#     seeds = 8
#     T = 200


#     def make_env(env_id, seed):
#         def _f():
#             env = gym.make(env_id)
#             env.seed(seed)
#             return env
#         return _f
    
#     envs = [make_env(env_name, seed) for seed in range(seeds)]
#     envs = DummyVecEnv(envs)

#     obs = envs.reset()

#     for t in range(T):
#         actions = np.stack([envs.action_space.sample() for _ in range(seeds)])
#         obs, rewards, dones, infos = envs.step(actions)
#         envs.render()