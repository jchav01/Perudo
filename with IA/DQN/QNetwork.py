# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:54:31 2024

@author: jules
"""


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from player import Player 
import numpy as np



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)




class DQNAgent:
    def __init__(self, state_dim, max_actions, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.996, min_epsilon=0.01, replay_buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.max_actions = max_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.model = DQN(state_dim, max_actions)
        self.target_model = DQN(state_dim, max_actions)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def encode_action(self, action, max_b=6):
        a, b = action
        return a * 10 + b

    def decode_action(self, action_index, max_b=6):
        a = action_index // 10
        b = action_index % 10
        return a, b 

    def store_transition(self, state, action, reward, next_state, done):
        action_index = self.encode_action(action)
        self.replay_buffer.push(state, action_index, reward, next_state, done)

    def generate_action_space(self, state, num_players):
        action_space = [(0, 0)]
        total_dice = state[-1]* (5 * num_players)
        current_bid = self.decode_action(state[-2])

        for a in range(int(current_bid[0]), round(total_dice/2) + 2):
            if a == current_bid[0]:
                for b in range(int(current_bid[1]) + 1, 7):
                    action_space.append((a, b))
            else:
                for b in range(1, 7):
                    action_space.append((a, b))

        if current_bid == (1, 0):
            action_space.remove((0, 0))

        return action_space

    def choose_action(self, state, num_players):
        action_space = self.generate_action_space(state, num_players)
        action_indices = [self.encode_action(a) for a in action_space]

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(action_space)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor).detach().numpy()[0]
            valid_q_values = np.array([q_values[idx] for idx in action_indices])
            action_index = action_indices[np.argmax(valid_q_values)]
            action = self.decode_action(action_index)

        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Ensure actions have the correct shape
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
    
        

class DQNAgentWrapper(Player):
    def __init__(self, name, dqn_agent):
        super().__init__(name, strategy='dqn')
        self.dqn_agent = dqn_agent
        self.proposed_bid = None

    def bet(self, current_bid, total_dice):
        return self.proposed_bid
    
    def set_proposed_bid(self, proposed_bid):
        self.proposed_bid = proposed_bid
