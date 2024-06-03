# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:53:07 2024

@author: jules
"""
import random
from player import Player
import pickle

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.80, epsilon=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
    
    def encode_state(self, dice, current_bid, total_dice):
        organized = self.organized_dice(dice)
        return (tuple(organized), current_bid, total_dice)
    
    def organized_dice(self, dice):
        dice_count = [0] * 6
        for die in dice:
            dice_count[die - 1] += 1
        organized = [(count, face + 1) for face, count in enumerate(dice_count)]
        return organized

    def encode_action(self, action):
        return action

    def decode_action(self, action_index):
        return action_index

    def initialize_state(self, state, action_space):
        if state not in self.q_table:
            self.q_table[state] = {self.encode_action(a): 0 for a in action_space}

    def choose_action(self, state, action_space):
        self.initialize_state(state, action_space)
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(action_space)
        else:
            q_values = self.q_table[state]
            max_q_value = max(q_values.values())
            best_actions = [a for a in action_space if q_values[self.encode_action(a)] == max_q_value]
            action = random.choice(best_actions)

        return self.encode_action(action)

    def update_q_table(self, state, action_index, reward, next_state, next_action_space):
        self.initialize_state(next_state, next_action_space)
        
        current_q_value = self.q_table[state].get(action_index, 0)
        max_next_q_value = max(self.q_table[next_state].values(), default=0)

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_table[state][action_index] = new_q_value

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

class QLearningAgentWrapper(Player):
    def __init__(self, name, q_agent):
        super().__init__(name, strategy='q-learning')
        self.q_agent = q_agent

    def bet(self, current_bid, total_dice):
        return self.proposed_bid

    def set_proposed_bid(self, proposed_bid):
        self.proposed_bid = proposed_bid