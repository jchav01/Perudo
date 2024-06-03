# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:54:31 2024

@author: jules
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from player import Player

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 1000

        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma *
                          torch.max(self.target_model(next_state)[0]).item())
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach()
            target_f[0][action] = target
            target_f = target_f.unsqueeze(0)

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state).unsqueeze(0))
            loss = self.loss_fn(output, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgentWrapper(Player):
    def __init__(self, name, dqn_agent):
        super().__init__(name, strategy='dqn')
        self.dqn_agent = dqn_agent

    def bet(self, current_bid, total_dice):
        return self.proposed_bid
    
    def set_proposed_bid(self, proposed_bid):
        self.proposed_bid = proposed_bid
