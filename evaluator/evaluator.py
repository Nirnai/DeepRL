import os
import shutil
import time
import gym
import numpy as np
from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from itertools import count

class Evaluator():
    def __init__(self, algorithm, path):
        self.alg = algorithm
        self.eval_alg = deepcopy(self.alg)
        self.param = algorithm.param
        self.alg_name = algorithm.name
        # self.env_name = algorithm.env.spec.id
        self.env_name = algorithm.env.spec._kwargs['domain_name'] + algorithm.env.spec._kwargs['task_name']
        self.out_dir = self.create_output_dir(path)
        # Parameters
        self.total_timesteps = int(self.param.evaluation['total_timesteps'])
        self.eval_timesteps = int(self.param.evaluation['eval_timesteps'])
        self.eval_episodes = int(self.param.evaluation['eval_episodes'])
        self.window = int(self.param.evaluation['avaraging_window'])
        self.desired_average_return = algorithm.env.spec.reward_threshold
        self.log_interval = 1
        # Metrics
        self.curr_episode = 0
        self.returns = [0.0]
        self.average_returns_online = []
        self.average_returns_offline = []
        self.deviation_returns_offline = []
        self.final_returns = []
        self.final_deviation = []
        self.robust_returns = []
        self.robust_deviation = []
        self.metrics = dict()
        # Counter
        self.sample = 0


    def run_statistic(self, samples=10, seed=100):
        self.param.save_parameters(self.out_dir)
        seeds = []
        rng = np.random.RandomState(seed)
        for i in range(samples):
            seeds.append(rng.randint(0,100))
            self.sample = i
            self.reset()
            self.run()
            # Save Data
            self.save_policy()
            self.save_returns()
            self.save_metrics()

    def run(self):
        done = True
        for t in range(self.total_timesteps):
            if done:
                state = self.alg.env.reset()
            # Act
            state, reward, done = self.alg.act(state)
            # Learn
            metrics = self.alg.learn()
            # Collect Metrics
            self.log_metrics(metrics)
            # Collect Returns
            self.log_reward(reward, done)
            # Evaluate 
            self.eval_progress()
            # Print Info
            if done and self.curr_episode % self.log_interval == 0:
                self.print_progress()
    
    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def log_reward(self, reward, done):
        if done:
            self.curr_episode += 1
            self.returns.append(0.0)
            self.average_returns_online.append(np.mean(self.returns[-self.window:-1]))
        else:
            self.returns[-1] += reward

    def log_metrics(self, metrics):
        if type(metrics) is dict:
            for key, value in metrics.items():
                if value is not None:
                    if key in self.metrics:
                        self.metrics[key].append(value)
                    else:
                        self.metrics[key] = [value]

    def eval_progress(self):
        if self.alg.steps % self.eval_timesteps == 0:
            self.eval_alg.actor = deepcopy(self.alg.actor)
            returns = []
            for episode in range(self.eval_episodes):
                state = self.eval_alg.env.reset()
                r = 0
                while True:
                    state, reward, done = self.eval_alg.act(state, deterministic=True)
                    r += reward
                    if done:
                        break
                returns.append(r)
            self.average_returns_offline.append(np.mean(returns)) 
            self.deviation_returns_offline.append(np.std(returns))             


    def eval_policy(self):
        # self.eval_alg.actor.load_state_dict(self.alg.actor.state_dict())
        self.eval_alg.actor = deepcopy(self.alg.actor)
        returns = []
        for episode in range(self.eval_episodes):
            state = self.eval_alg.env.reset()
            r = 0
            for t in count():
                state, reward, done = self.eval_alg.act(state, deterministic=True)
                r += reward
                if done:
                    break
            returns.append(r)
        self.final_returns.append(np.mean(returns)) 
        self.final_deviation.append(np.std(returns))   


    def eval_robustness(self):
        high = np.ones(2) * self.param.evaluation['max_external_force']
        low = -high
        dist_space = gym.spaces.Box(low, high)
        self.eval_alg.actor = deepcopy(self.alg.actor)
        returns = []
        for episode in range(self.eval_episodes):
            state = self.eval_alg.env.reset()
            r = 0
            for t in count():
                if t%self.param.evaluation['disturbance_intervall'] == 0:
                    ## Disturbance ##
                    dist = dist_space.sample()
                    self.eval_alg.env.env.physics.data.xfrc_applied[2][0] = dist[0]
                    self.eval_alg.env.env.physics.data.xfrc_applied[2][2] = dist[1]
                    #################
                state, reward, done = self.eval_alg.act(state, deterministic=True)
                r += reward
                if done:
                    break
            returns.append(r)
        self.robust_returns.append(np.mean(returns)) 
        self.robust_deviation.append(np.std(returns))   

    ## Saving
    def create_output_dir(self, base):
        timestr = time.strftime("%Y-%m-%d_%H-%M")
        dir_name = self.alg_name + '_' + self.env_name + '_' + timestr
        directory = base + '/' + dir_name
        if not os.path.isdir(directory):  
            os.makedirs(directory, exist_ok=True)
        return directory
    
    def save_policy(self):
        self.alg.save_model(self.out_dir)
        path = self.out_dir + '/policies'
        if not os.path.isdir(path):  
            os.mkdir(path)
        shutil.copyfile(self.out_dir + '/actor_model.pt', path + '/actor_model_{}.pt'.format(self.sample))

    def save_returns(self):
        returns_file_online = '{}/returns_online.npz'.format(self.out_dir)
        returns_file_offline = '{}/returns_offline.npz'.format(self.out_dir)
        deviation_file_offline = '{}/deviation_offline.npz'.format(self.out_dir)
        # finalreturns_file = '{}/final_returns.npz'.format(self.out_dir)
        # finaldeviation_file = '{}/final_deviation.npz'.format(self.out_dir)
        # robustreturns_file = '{}/robust_returns.npz'.format(self.out_dir)
        # robustdeviation_file = '{}/robust_deviation.npz'.format(self.out_dir)

        returns_files = [returns_file_online, 
                         returns_file_offline, 
                         deviation_file_offline]

        returns = [self.average_returns_online, 
                   self.average_returns_offline, 
                   self.deviation_returns_offline]

        for fil, ret in zip(returns_files, returns):
            samples = []
            if os.path.isfile(fil):
                samples = [array for array in np.load(fil).values()]
            samples.append(ret)
            np.savez(fil[:-4], *samples)
    
    def save_metrics(self):
        for key, values in self.metrics.items():
            key.replace(" ", "")                
            filename = '{}/{}.npz'.format(self.out_dir, key)
            samples = []
            if os.path.isfile(filename):
                samples = [array for array in np.load(filename).values()]
            samples.append(values)
            np.savez(filename[:-4], *samples)
    
    def save_video(self, disturbance=False):
        done = True
        frames = []
        if disturbance:
            path = self.out_dir + '/video_robust'
            high = np.ones(2) * self.param.evaluation['max_external_force']
            low = -high
            dist_space = gym.spaces.Box(low, high)
        else:
            path = self.out_dir +'/video'
        for t in range(1000):
            if done:
                state = self.alg.env.reset()
            if disturbance:
                # Disturbance ##
                dist = dist_space.sample()
                self.alg.env.env.physics.data.xfrc_applied[2][0] = dist[0]
                self.alg.env.env.physics.data.xfrc_applied[2][2] = dist[1]
                #################
            state, reward, done = self.alg.act(state, deterministic=True)
            frames.append(self.alg.env.render(mode='rgb_array'))
        fig = plt.figure()
        im = []
        for frame in frames:
            im.append([plt.imshow(frame)])
        ani = animation.ArtistAnimation(fig, im, interval=10, blit=True, repeat_delay=1000)
        ani.save('{}.mp4'.format(path))


    def reset(self):
        self.curr_episode = 0
        self.returns = [0.0]
        self.average_returns_online = []
        self.average_returns_offline = []
        self.deviation_returns_offline = []
        self.final_returns = []
        self.final_deviation = []
        self.robust_returns = []
        self.robust_deviation = []
        self.metrics = dict()
        self.solved = False
        self.alg.reset()


    def print_progress(self):
        print("Steps: {:,}".format(self.alg.steps))
        print("Episode: {:.2f}".format(self.curr_episode))
        print("Average Return: {:.2f}".format(self.average_returns_online[-1]))
        print("Goal Average Return: {}".format(self.desired_average_return))
        for key, value in self.metrics.items():
            print("{}: {:.6f}".format(key, self.metrics[key][-1]))
        print("------------------------------------")
    

    def seed(self, seed):
        pass
        # self.alg.env.seed(seed)
        # self.alg.seed(seed)


