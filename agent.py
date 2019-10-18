import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

MODEL_PATH = '/Users/Hugh/Documents/Coding/Reinforcement Learning/Open AI Gym Tests/Cart Pole'


class CartPoleAgent:

    def __init__(self, env):
        print('Agent Initialized')
        self.environment = env
        self.model = NeuralNetwork()

    def act(self, obs):
        action = self.model.forward_pass(obs)
        assert self.environment.action_space.contains(action)
        return action

    def save_model(self):
        torch.save(self.model, MODEL_PATH)

    def load_model(self):
        self.model = torch.load(MODEL_PATH)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dense1 = nn.Linear()
        self.dense2 = nn.Linear()
        self.dense3 = nn.Linear()

    def forward_pass(self, x):
        x = f.relu(self.dense1(x))
        x = f.relu(self.dense2(x))
        x = f.relu(self.dense3(x))
        return x


