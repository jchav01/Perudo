# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:53 2024

@author: jules
"""

from player import Player
from QNetwork import DQNAgent, DQNAgentWrapper
import numpy as np

class CustomGameEnv:
    def __init__(self, num_players, DQN_agent):
        self.num_players = num_players
        self.players = self.create_players(num_players, DQN_agent)
        self.current_bid = None
        self.DQN_agent = DQN_agent
        self.state = self.reset()
        
    def reset(self, num_players):
        organized_dice = [(0, n+1) for n in range(6)]
        total_dice = self.num_players * 6
        starting_bid = (1, 0)
        self.state = (organized_dice, starting_bid, total_dice)
        
        self.players = self.create_players(self.num_players, self.DQN_agent)
        
        return self.state

    def create_players(self, num_players, DQN_agent):
        players = []
        strategies = ['prudent', 'brusque']
        for i in range(1, num_players):
            strategy = strategies[(i - 1) % len(strategies)]
            players.append(Player(f"Player {i}", strategy))
        
        if DQN_agent:
            players.append(DQNAgentWrapper(f"Player {num_players}", DQN_agent))
        
        return players

    def generate_action_space(self, current_bid, total_dice):
        action_space = [(0, 0)]
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

    def start(self):
        
        loser_index = 0
        
        score_table = []
        
        total_reward = 0
        
        sum_of_rewards = 0
        
        while len([p for p in self.players if p.has_dice()]) > 1:
            score_table, loser_index, sum_of_rewards = self.play_round(loser_index, score_table, sum_of_rewards)
        
        total_reward = sum_of_rewards
        
        for p in self.players:
            if p.has_dice():
                score_table.append(self.players.index(p))
  
        return score_table, total_reward

    def show_players_dice(self):
        print("\n Current dice of all players:")
        for player in self.players:
            if player.has_dice():
                print(f"{player.name}: {player.get_dice()} ({player.organize_dice()})")

    def count_total_dice(self):
        return sum(len(player.get_dice()) for player in self.players if player.has_dice())

    def count_active_players(self):
        return sum(1 for player in self.players if player.has_dice())

    def play_round(self, loser_index, score_table, sum_of_rewards):
        for player in self.players:
            if player.has_dice():
                player.roll_dice()
                
        self.show_players_dice()
        
        proposal = (1, 0)
        self.current_bid = proposal
        
        total_dice = self.count_total_dice()
        previous_index = None
        i = loser_index
    
        while True:
            current_player = self.players[i]
            
            if current_player.has_dice():
                
                if isinstance(current_player, DQNAgentWrapper):
                    action_space = self.generate_action_space(self.current_bid, total_dice)
                    state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                    action_index = self.DQN_agent.choose_action(state, action_space)
                    proposal = self.DQN_agent.decode_action(action_index)
                    
                    if proposal not in action_space:
                        raise ValueError(f"Q-agent chose an invalid bid: {proposal}")
                    
                else:
                    proposal = current_player.bet(self.current_bid, total_dice)
                    
                    
                if proposal != (0, 0):
                    self.current_bid = proposal
                    
                    if previous_index != None and isinstance(self.players[previous_index], DQNAgentWrapper):
                        reward = 25
                        sum_of_rewards += reward
                        #print("passed a tour ! reward:", reward)
                        next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                        next_action_space = self.generate_action_space(self.current_bid, total_dice)
                        self.DQN_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                else:
                    if previous_index == None:
                        raise ValueError(f"{current_player.name} said myth from start")
                    break
    
                previous_index = i
                
                print(f"{current_player.name} ({current_player.strategy})  fait un pari: {self.current_bid}")

    
            i = (i + 1) % len(self.players)
    
        index1 = previous_index
        index2 = self.players.index(current_player)
        
        
        count = 0
        for k in range(len(self.players)):
            for l in range(len(self.players[k].dice)):
                if self.players[k].dice[l] == self.current_bid[1]:
                    count += 1
        
        
        print(f"{self.players[index2].name} challenges {self.players[index1].name}: {self.current_bid}")
        print(f"result: ({count}, {self.current_bid[1]})") 
        
        """
        if isinstance(current_player, DQNAgentWrapper) or isinstance(self.players[index1], DQNAgentWrapper):
            print(f"{self.players[index2].name} challenges {self.players[index1].name}: {self.current_bid}")
            print(f"result: ({count}, {self.current_bid[1]})")    
        """
                
        reward = 0
        if count >= self.current_bid[0]:
            
            self.players[index2].lose_die()
            print(f"{self.players[index2].name} lost one die")
            if isinstance(current_player, DQNAgentWrapper):
                reward = -(count - self.current_bid[0]) ** 2
                sum_of_rewards += reward
                next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.DQN_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            if isinstance(self.players[index1], DQNAgentWrapper):
                reward = int(100 * np.exp(-(count - self.current_bid[0]))) 
                next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.DQN_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            
            if reward != 0:
                print("reward:", reward)
            
            
            if not self.players[index2].has_dice():
                score_table.append(index2)
                print(f"{self.players[index2].name} is eliminated")
                
            return score_table, index2, sum_of_rewards
        
        else:
            
            self.players[index1].lose_die()
            print(f"{self.players[index1].name} lost one die")
            if isinstance(self.players[index1], DQNAgentWrapper):
                reward = -(count - self.current_bid[0])**2
                sum_of_rewards += reward
                next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.DQN_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            if isinstance(current_player, DQNAgentWrapper):
                reward = (self.current_bid[0] - count)**2 + 10
                sum_of_rewards += reward
                next_state = self.DQN_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.DQN_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
               
            if reward != 0:
                print("reward:", reward)
            
            if not self.players[index1].has_dice():
                score_table.append(index1)
                print(f"{self.players[index1].name} is eliminated")
                
            return score_table, index2, sum_of_rewards