import os
import pytest
import torch
import yaml
from algorithms.hyperparameter import HyperParameter

def test_json_parser():
    save_param = HyperParameter()
    save_param.NETWOR_ARCHITECTURE = [4,10,2]
    save_param.MEMORY_SIZE = 1000
    save_param.BATCH_SIZE = 32
    save_param.EPSILON = 1
    save_param.EPSILON_MAX = 1.0
    save_param.EPSILON_MIN = 0.01
    save_param.EPSILON_DECAY = 500
    save_param.GAMMA = 0.99
    save_param.TARGET_UPDATE = 100
    save_param.LEARNING_RATE = 0.001
    save_param.save_parameters('test/test.json')
    assert(os.path.isfile('test/test.json'))
    load_param = HyperParameter()
    load_param.load_parameters('test/test.json')
    assert(load_param != save_param)
    os.remove('test/test.json')
    # print("saved parameters : {} \n  loaded parameters : {}".format(save_param, load_param))

# def test_yaml_parser():
#     param = HyperParameter()
#     param.NETWOR_ARCHITECTURE = [4,10,2]
#     param.MEMORY_SIZE = 1000
#     param.BATCH_SIZE = 32

#     with open("data.yaml", 'r') as stream:
#         obj = yaml.full_load(stream)
#     print(obj)