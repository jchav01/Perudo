# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:53 2024

@author: jules
"""

from player import Player
from Qlearning import QLearningAgentWrapper
import numpy as np

class Game:
    def __init__(self, num_players, q_agent):
        self.players = self.create_players(num_players, q_agent)
        self.current_bid = None
        self.q_agent = q_agent

    def create_players(self, num_players, q_agent):
        players = []
        strategies = ['prudent', 'brusque']
        for i in range(1, num_players):
            strategy = strategies[(i - 1) % len(strategies)]
            players.append(Player(f"Player {i}", strategy))
        
        if q_agent:
            players.append(QLearningAgentWrapper(f"Player {num_players}", q_agent))
        
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
        loser_indice = 0
        score_table = []
        total_reward = 0
        sum_of_rewards = 0
        while len([p for p in self.players if p.has_dice()]) > 1:
            score_table, loser_indice, sum_of_rewards = self.play_round(loser_indice, score_table, sum_of_rewards)
        
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

    def play_round(self, loser_indice, score_table, sum_of_rewards):
        for player in self.players:
            if player.has_dice():
                player.roll_dice()
                
        self.show_players_dice()
        
        proposal = (1, 0)
        self.current_bid = proposal
        
        total_dice = self.count_total_dice()
        previous_indice = None
        i = loser_indice
    
        while True:
            current_player = self.players[i]
            
            if current_player.has_dice():
                
                if isinstance(current_player, QLearningAgentWrapper):
                    action_space = self.generate_action_space(self.current_bid, total_dice)
                    state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                    action_index = self.q_agent.choose_action(state, action_space)
                    proposal = self.q_agent.decode_action(action_index)
                    
                    if proposal not in action_space:
                        raise ValueError(f"Q-agent chose an invalid bid: {proposal}")
                    
                else:
                    proposal = current_player.bet(self.current_bid, total_dice)
                    
                    
                if proposal != (0, 0):
                    self.current_bid = proposal
                    
                    if previous_indice != None and isinstance(self.players[previous_indice], QLearningAgentWrapper):
                        reward = 25
                        sum_of_rewards += reward
                        #print("passed a tour ! reward:", reward)
                        next_state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                        next_action_space = self.generate_action_space(self.current_bid, total_dice)
                        self.q_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                else:
                    if previous_indice == None:
                        raise ValueError(f"{current_player.name} said myth from start")
                    break
    
                previous_indice = i
                
                print(f"{current_player.name} ({current_player.strategy})  fait un pari: {self.current_bid}")

    
            i = (i + 1) % len(self.players)
    
        indice1 = previous_indice
        indice2 = self.players.index(current_player)
        
        
        count = 0
        for k in range(len(self.players)):
            for l in range(len(self.players[k].dice)):
                if self.players[k].dice[l] == self.current_bid[1]:
                    count += 1
        
        
        print(f"{self.players[indice2].name} challenges {self.players[indice1].name}: {self.current_bid}")
        print(f"result: ({count}, {self.current_bid[1]})") 
        
        """
        if isinstance(current_player, QLearningAgentWrapper) or isinstance(self.players[indice1], QLearningAgentWrapper):
            print(f"{self.players[indice2].name} challenges {self.players[indice1].name}: {self.current_bid}")
            print(f"result: ({count}, {self.current_bid[1]})")    
        """
                
        reward = 0
        if count >= self.current_bid[0]:
            
            self.players[indice2].lose_die()
            print(f"{self.players[indice2].name} lost one die")
            if isinstance(current_player, QLearningAgentWrapper):
                reward = -(count - self.current_bid[0]) ** 2
                sum_of_rewards += reward
                next_state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.q_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            if isinstance(self.players[indice1], QLearningAgentWrapper):
                reward = int(100 * np.exp(-(count - self.current_bid[0]))) 
                next_state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.q_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            
            if reward != 0:
                print("reward:", reward)
            
            
            if not self.players[indice2].has_dice():
                score_table.append(indice2)
                print(f"{self.players[indice2].name} is eliminated")
                
            return score_table, indice2, sum_of_rewards
        
        else:
            
            self.players[indice1].lose_die()
            print(f"{self.players[indice1].name} lost one die")
            if isinstance(self.players[indice1], QLearningAgentWrapper):
                reward = -(count - self.current_bid[0])**2
                sum_of_rewards += reward
                next_state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.q_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
            if isinstance(current_player, QLearningAgentWrapper):
                reward = (self.current_bid[0] - count)**2 + 10
                sum_of_rewards += reward
                next_state = self.q_agent.encode_state(current_player.get_dice(), self.current_bid, total_dice)
                next_action_space = self.generate_action_space(self.current_bid, total_dice)
                self.q_agent.update_q_table(state, action_index, reward, next_state, next_action_space)
                
               
            if reward != 0:
                print("reward:", reward)
            
            if not self.players[indice1].has_dice():
                score_table.append(indice1)
                print(f"{self.players[indice1].name} is eliminated")
                
            return score_table, indice2, sum_of_rewards