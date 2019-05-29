import torch
import gym
from algorithms import TRPO

def test_conjugate_gradient():
    A = torch.Tensor([[1,2,3], [2,-1,1], [3,0,-1]])
    b = torch.Tensor([9,8,3])
    x = torch.Tensor([2,-1,3])

    env = gym.make('Pendulum-v0')
    alg = TRPO(env)
    x_cg = alg.conjugate_gradient(A,b,100)

    mse = torch.mean((x - x_cg)**2)

    print('True Result x = {}'.format(x))
    print('CG Result x_cg = {}'.format(x_cg))
    print('MSE = {}'.format(mse))
    assert(mse < 1e-4)