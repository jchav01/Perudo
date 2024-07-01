# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:07:47 2024

@author: jules
"""

import random
import numpy as np
from collections import Counter
from dqn_agent import DQNAgent


class Player:
    def __init__(self, name, strategy = 'brusque'):
        self.name = name
        self.strategy = strategy
        self.dice = [random.randint(1, 6) for _ in range(5)]
        self.eliminated = False
        self.reward = 0
        self.score = -1

    def roll_dice(self):
        if not self.eliminated:
            self.dice = [random.randint(1, 6) for _ in range(len(self.dice))]

    def lose_die(self):
        if self.dice:
            self.dice.pop()
            if not self.dice:
                self.eliminate()

    def has_dice(self):
        return len(self.dice) > 0 and not self.eliminated

    def eliminate(self):
        self.eliminated = True

    def get_dice(self):
        return self.dice

    def organize_dice(self):
        dice_count = Counter(self.dice)
        organized_dice = [(count, value) for value, count in dice_count.items()]
        organized_dice.sort(key=lambda x: x[1])
        return organized_dice

    def best_bet(self):
        max_count = 0
        best = (0, 0)
        organized_dice = self.organize_dice()
        for k in organized_dice:
            if k[0] >= max_count:
                max_count = k[0]
                best = k
        return best

    def brusque_bet(self, current_bid, total_dice):
        
        self.proposed_bid = (0,0)
        
        best = self.best_bet()
        if current_bid[0] < best[0]:
            self.proposed_bid = (best[0], best[1])
        elif current_bid[0] <= best[0] + round(1/6*total_dice) and current_bid[1] < best[1]:
            self.proposed_bid = (current_bid[0] + 1, best[1])
        elif current_bid[0] < best[0] + round(1/6*total_dice) and current_bid[1] >= best[1]:
            self.proposed_bid = (current_bid[0] + 1, current_bid[1])
        
        return self.proposed_bid
    
    def get_state(self, current_bid, total_dice, num_players):

        total_dice_indicator = total_dice/(num_players * 5)
        dice = self.get_dice()
        dice_state = [0] * 6
        for die in dice:
            dice_state[die - 1] += 1
    
        encoded_bid = self.encode_action(current_bid)
        
        
        return np.array(dice_state + [encoded_bid] + [total_dice_indicator])
    
    def encode_action(self, action):
        a, b = action
        return a * 10 + b

class DQNAgentWrapper(Player):
    def __init__(self, name, dqn_agent):
        super().__init__(name, strategy = 'dqn')
        self.dqn_agent = dqn_agent
        self.proposed_bid = None

        
    def bet(self, current_bid, total_dice, num_players, bid_history):
        state = self.get_state(current_bid, total_dice, bid_history)
        action_index = DQNAgent.choose_action(state)
        self.proposed_bid = self.decode_action(action_index)
        return self.proposed_bid

    def get_state(self, current_bid, total_dice, num_players):

        total_dice_indicator = total_dice/(num_players * 5)
        dice = self.get_dice()
        dice_state = [0] * 6
        for die in dice:
            dice_state[die - 1] += 1
            
        encoded_bid = self.encode_action(current_bid)
    
        return np.array(dice_state + [encoded_bid] + [total_dice_indicator])
        
    def encode_action(self, action):
        a, b = action
        return a * 10 + b

    def decode_action(self, action_index):
        a = action_index // 10
        b = action_index % 10
        return a, b


        