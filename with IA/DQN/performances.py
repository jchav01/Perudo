# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:51:03 2024

@author: jules
"""

import matplotlib.pyplot as plt
import numpy as np

class PerformanceTracker:
    def __init__(self):
        self.positions = []
        self.rewards = []
        self.cumulative_rewards = []
        self.epsilon_values = []

    def record_position(self, position):
        self.positions.append(position)

    def record_reward(self, reward):
        self.rewards.append(reward)
        if len(self.cumulative_rewards) == 0:
            self.cumulative_rewards.append(reward)
        else:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)

    def record_epsilon(self, epsilon):
        self.epsilon_values.append(epsilon)

    def plot_positions(self, num_players):
        
        all_positions = np.arange(1, num_players + 1)
        positions, counts = np.unique(self.positions, return_counts=True)
        
        all_counts = np.zeros_like(all_positions)
        for pos, count in zip(positions, counts):
            all_counts[pos] = count
    
        plt.bar(all_positions, all_counts)
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title('Q-learning Agent Position Distribution')
        plt.xticks(all_positions)
        plt.show()
    
    def plot_rewards(self):
        plt.plot(self.rewards)
        plt.xlabel('Games')
        plt.ylabel('Rewards')
        plt.title('Rewards over Time')
        plt.show()
        
    def plot_cumulative_rewards(self):
        plt.plot(self.cumulative_rewards)
        plt.xlabel('Games')
        plt.ylabel('Cumulative Rewards')
        plt.title('Cumulative Rewards over Time')
        plt.show()

    def plot_epsilon_decay(self):
        plt.plot(self.epsilon_values)
        plt.xlabel('Games')
        plt.ylabel('Epsilon Value')
        plt.title('Epsilon Decay over Time')
        plt.show()



