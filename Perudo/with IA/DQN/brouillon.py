# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:28:03 2024

@author: PC
"""


sum_of_rewards += reward
next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)

if isinstance(current_player, DQNAgentWrapper):
    reward = -(count - self.current_bid[0]) ** 2
    self.DQN_agent.store_transition(state, proposal, reward, next_state, done)
    self.DQN_agent.learn()
    
if isinstance(self.players[index1], DQNAgentWrapper):
    reward = int(100 * np.exp(-(count - self.current_bid[0]))) 
    next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
    self.DQN_agent.store_transition(state, proposal, reward, next_state, done)
    self.DQN_agent.learn()
    
if isinstance(self.players[index1], DQNAgentWrapper):
    reward = -(count - self.current_bid[0])**2
    sum_of_rewards += reward
    next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
    self.DQN_agent.store_transition(state, proposal, reward, next_state, done)
    self.DQN_agent.learn()
    
if isinstance(current_player, DQNAgentWrapper):
    reward = (self.current_bid[0] - count)**2 + 10
    sum_of_rewards += reward
    next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
    self.DQN_agent.store_transition(state, proposal, reward, next_state, done)
    self.DQN_agent.learn()