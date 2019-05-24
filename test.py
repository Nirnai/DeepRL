import torch
import numpy
import gym
from algorithms import TRPO

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    alg = TRPO(env)
    
    mean1 = torch.Tensor([0])
    mean2 = torch.Tensor([1])
    std1 = torch.Tensor([0.1])
    std2 = torch.Tensor([0.3])

    d_kl = torch.log(std2) - torch.log(std1) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 1/2

    print(d_kl)

    p = torch.distributions.Normal(mean1, std1)
    q = torch.distributions.Normal(mean2, std2)

    # x = p.sample((100,))
    x = numpy.linspace(-10,10,100)
    x = torch.Tensor(x)

    print(torch.distributions.kl.kl_divergence(p,q))

    d_kl = torch.distributions.kl.kl_divergence(p,q)

    v = torch.ones(3)

    alg.hessien_vector(d_kl, v)


    # print(alg.param)