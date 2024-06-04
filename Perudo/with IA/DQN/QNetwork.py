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



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

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
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_buffer_size=10000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    
    def choose_action(self, state):
        action_space = self.generate_action_space(state)
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(action_space)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor).detach().numpy()[0]
    
            # Filter Q-values to only consider those in the action_space
            valid_q_values = {action: q_values[self.encode_action(action)] for action in action_space}
            
            # Ensure there are valid actions to choose from
            if not valid_q_values:
                raise ValueError("No valid actions available")
            
            max_q_value = max(valid_q_values.values())
            best_actions = [a for a in action_space if valid_q_values[self.encode_action(a)] == max_q_value]
            action = random.choice(best_actions)
            
        if action not in action_space:
            raise ValueError(f"Chosen action {action} is not in the valid action space {action_space}")
        
        return action

    def encode_action(self, action):
        # Convert action (tuple) to an integer index if necessary
        return action[0] * 10 + action[1]  # Example encoding function, adjust as needed
    
    def decode_action(self, action):
        return(int(action/10), action - 10*int(action/10))
               
    def generate_action_space(self, state):
        action_space = [(0, 0)]
        
        total_dice = state[-1]
        current_bid = state[-2]
        current_bid = self.decode_action(current_bid)
        
        for a in range(current_bid[0], round(total_dice / 2) + 2):
            if a == current_bid[0]:
                for b in range(current_bid[1] + 1, 7):
                    action_space.append((a, b))
            else:
                for b in range(1, 7):
                    action_space.append((a, b))
        if current_bid == (1, 0):
            action_space.remove((0, 0))
        
        return action_space

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
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

    def initialize_state(self, state, action_space):
        if state not in self.q_table:
            self.q_table[state] = {self.encode_action(a): 0 for a in action_space}

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class DQNAgentWrapper(Player):
    def __init__(self, name, dqn_agent):
        super().__init__(name, strategy='dqn')
        self.dqn_agent = dqn_agent
        self.proposed_bid = None

    def bet(self, current_bid, total_dice):
        return self.proposed_bid
    
    def set_proposed_bid(self, proposed_bid):
        self.proposed_bid = proposed_bid
