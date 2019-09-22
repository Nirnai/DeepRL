
import os
import numpy as np 
import matplotlib.pyplot as plt
from evaluator.statistics import mean_confidance

def load_dataset(path):
    if os.path.isfile(path):
            data = np.array([array for array in np.load(path).values()])
            return data
    else:
        raise FileNotFoundError

def plot(path, x=None):
    data = load_dataset(path)
    plt.figure()
    for sample in data:
        if x is None:
            plt.plot(sample)
        else:
            plt.plot(x,sample)

def plot_dataset(path, total_steps=1e6, goal=0 ,statistic=None, show=True):
    data = load_dataset(path)
    episodes = range(len(data[0]))
    fig = plt.figure()
    if statistic is None:
        for sample in data:
            plt.plot(episodes, sample)
        plt.plot(episodes, [goal] * len(episodes), 'k--', label = 'goal reward')
    elif statistic is 'normal':
        mean, low, high = mean_confidance(data)
        plt.plot(episodes, mean, label = 'mean')
        plt.fill_between(episodes, low, high, facecolor='lightblue', label='std-error')
        plt.plot(episodes, [goal] * len(episodes), 'k--', label = 'goal reward')
        plt.legend()
    else:
        NotImplementedError
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    if show:
        plt.show()
    return fig


def compare_datasets(paths, goal=0, show=True):
    fig = plt.figure()
    for path in paths:
        name = os.path.basename(path)[:-4]
        alg_name = name.split('_')[0]
        env_name = name.split('_')[1]
        if len(name.split('_')) == 3:
            exp_name = name.split('_')[2]
        else:
            exp_name = ''
        dataset = load_dataset(path)
        episodes = range(len(dataset[0]))
        mean, low, high = mean_confidance(dataset)
        plt.plot(episodes, mean, label='{} {}'.format(alg_name, exp_name))
        plt.fill_between(episodes, low, high, alpha=0.2)
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('Average Reward')
        plt.title('{}'.format(env_name))
    plt.plot(episodes, [goal] * len(episodes), 'k--', label = 'goal reward')
    if show:
        plt.show()
    return fig


def plot_action(path, env ,statistic=None, show=True):
    data = load_dataset(path)
    steps = range(len(data[0]))
    fig = plt.figure()
    for sample in data:
        plt.plot(steps, sample)
    plt.plot(steps, [env.action_space.high] * len(steps), 'k--')
    plt.plot(steps, [env.action_space.low] * len(steps), 'k--')
    plt.xlabel('Steps')
    plt.ylabel('Control')
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    fig = plot_dataset('example.npz', statistic='normal')
    plt.show()
