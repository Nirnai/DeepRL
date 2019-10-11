
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

def plot_dataset(path, total_steps=1e6, goal=0, statistic=None, show=True):
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

def plot_offline(path_returns, path_deviations, n):
    fig = plt.figure()
    for path_return, path_deviation in zip(path_returns, path_deviations):
        means = load_dataset(path_return)
        stds = load_dataset(path_deviation)
        n_total = len(means) * n
        new_mean = np.mean(means, axis=0)
        new_std = np.sqrt(n/n_total * (np.sum(stds**2, axis=0) + np.sum((means - new_mean)**2, axis=0)))
        new_se = new_std / np.sqrt(10)
        episodes = np.linspace(0, 1000 , len(new_mean))
        plt.plot(episodes, [1000] * len(episodes), 'k--', label = 'goal reward')
        plt.plot(episodes, new_mean)
        plt.fill_between(episodes, new_mean - new_std, new_mean + new_std, alpha=0.2)
    # plt.fill_between(range(len(new_mean)),new_mean - 1.96 * new_se, new_mean + 1.96 * new_se, alpha=0.5)


def plot_final_performance(paths_returns, paths_deviations, n):
    x = []
    means = []
    stds = []
    for path_returns, path_deviations in zip(paths_returns, paths_deviations):
        label = path_returns.split('/')[-2]
        mean = load_dataset(path_returns)
        std = load_dataset(path_deviations)
        n_total = len(mean) * n
        new_mean =  n/n_total * np.sum(mean)
        new_std = np.sqrt(n/n_total * (np.sum(std**2) + np.sum((mean - new_mean)**2)))
        x.append(label)
        means.append(new_mean)
        stds.append(new_std)
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)


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
        episodes = np.linspace(0,1000,len(dataset[0])) #range(len(dataset[0]))
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



    ################################################################
    ########################## Utilities ###########################
    ################################################################

    def combine_statistics(means, stds, n):
        n_total = len(means) * n
        new_mean = np.mean(means)
        new_std = np.sqrt(n/n_total * (np.sum(stds**2) + np.sum((means - new_mean)**2)))
        return new_mean, new_std
        


if __name__ == '__main__':
    fig = plot_dataset('example.npz', statistic='normal')
    plt.show()
