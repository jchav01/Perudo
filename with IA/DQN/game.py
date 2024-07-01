# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:53 2024

@author: jules
"""

from player import Player
from QNetwork import DQNAgentWrapper
import numpy as np

class CustomGameEnv:
    def __init__(self, num_players, DQN_agent_wrapper):
        self.num_players = num_players
        self.players = self.create_players(num_players, DQN_agent_wrapper)
        self.DQN_agent = DQN_agent_wrapper
        self.agent_index = None
        self.game_state = self.reset()
        self.total_dice = num_players * 5
        self.initial_total_dice = num_players * 5
        self.current_reward = 0
        self.current_bid = (1,0)
        self.previous_index = None
        self.current_index = None
        self.new_round = True
        self.done = False
        self.score = -1
        
    def reset(self):
        self.new_round = True
        self.done = False
        self.current_index = 0
        self.score = -1
        
        self.players = self.create_players(self.num_players, self.DQN_agent)
        
        for idx, p in enumerate(self.players):
            if isinstance(p, DQNAgentWrapper):
                self.agent_index = idx
        
        for player in self.players:
            player.eliminated = False
            
        self.total_dice = (self.num_players) * 5
        
    
    def set_score(self):
        count = 0
        for p in self.players:
            if p.has_dice():
                count += 1
        return count + 1
                

    def create_players(self, num_players, DQN_agent_wrapper):
        players = []
        strategies = ['prudent', 'brusque']
        for i in range(1, num_players):
            strategy = strategies[(i - 1) % len(strategies)]
            players.append(Player(f"Player {i}", strategy))
        
        if DQN_agent_wrapper:
            players.append(DQN_agent_wrapper)
    
        
        for player in players:
            player.dice = player.reset_dice()
        
        return players
    
    def encode_action(self, action, max_b=6):
        a, b = action
        return a * 10 + b

    def decode_action(self, action_index, max_b=6):
        a = action_index // 10
        b = action_index % 10
        return a, b
    
    def get_agent_state(self):
        current_bid = self.current_bid
        total_dice_indicator = self.total_dice/(self.initial_total_dice)
        dice = self.DQN_agent.get_dice()
        dice_state = [0] * 6
        for die in dice:
            dice_state[die - 1] += 1
    
        encoded_bid = self.encode_action(current_bid)
    
        # Create the state list
        return np.array(dice_state + [encoded_bid] + [total_dice_indicator])
    

    def show_players_dice(self):
        print("\n Current dice of all players:")
        for player in self.players:
            if player.has_dice():
                print(f"{player.name}: {player.get_dice()} {player.organize_dice()}")

    def count_total_dice(self):
        return sum(len(player.get_dice()) for player in self.players if player.has_dice())

    def count_active_players(self):
        return sum(1 for player in self.players if player.has_dice())
    
    def set_new_round(self):
        self.current_bid = (1,0)
        
        for player in self.players:
            if player.has_dice():
                player.roll_dice()
                
        #self.show_players_dice()
        
        self.previous_index = None
        
        
    def play_round(self):
         
        i = self.current_index
        
        current_player = self.players[i]
        
        if isinstance(current_player, DQNAgentWrapper) and self.new_round == True:
            self.set_new_round()
            self.new_round = False
        
        while not isinstance(current_player, DQNAgentWrapper) and self.done == False:
                
            if current_player.has_dice():
                
                proposal = current_player.bet(self.current_bid, self.total_dice)
                
                if proposal == (0, 0):
                    if self.previous_index == None:
                        raise ValueError(f"{current_player.name} said myth from start")
                    else:
                        starting_index = self.checking()
                        self.new_round = True
                        self.current_index = starting_index

                else:
                    self.current_bid = proposal
                    #print(f"{current_player.name} ({current_player.strategy})  fait un pari: {self.current_bid}")
                    self.previous_index = i

                    if self.previous_index and isinstance(self.players[self.previous_index], DQNAgentWrapper):
                        self.current_reward += 5*self.current_bid[0]
                
            if self.done == True:
                break
            
            if self.new_round == True:
                self.set_new_round()
                self.new_round = False
                current_player = self.players[self.current_index]
                
            else:
                i = (i + 1) % len(self.players)
                self.current_index = i
                current_player = self.players[self.current_index]
                
        
        # On sors de la boucle car c'est à l'agent de jouer ou car c'est la fin du jeu
        state = self.get_agent_state()
        #print(state)
        return state
    
    def checking(self):
        
        index1 = self.previous_index
        index2 = self.current_index
        
        current_player = self.players[index2]
        
        count = 0
        
        for k in range(len(self.players)):
            for l in range(len(self.players[k].dice)):
                if self.players[k].dice[l] == self.current_bid[1]:
                    count += 1
            
        #print(f"{self.players[index2].name} challenges {self.players[index1].name}: {self.current_bid}")
        #print(f"result: ({count}, {self.current_bid[1]})")
    
        
        self.total_dice -= 1
        
        if count >= self.current_bid[0]:
            
            self.players[index2].lose_die()
            #print(f"{self.players[index2].name} lost one die")
            
            if isinstance(current_player, DQNAgentWrapper):
                self.current_reward += -(count - self.current_bid[0]) ** 2 - 25
                
         
            if not self.players[index2].has_dice():
                #score_table.append(index2)
                #print(f"{self.players[index2].name} is eliminated")
                if not self.DQN_agent.has_dice():
                    self.done = True
                    self.score = self.set_score()
            
            
            
            if len([p for p in self.players if p.has_dice()]) == 1:
                if self.DQN_agent.has_dice():
                    self.current_reward += 350
                    self.score = 1
                    #print("Won the game ! reward:100")
                    
                self.done = True
                if self.score != 1:
                    self.score = self.set_score()
            
            return index2
        
        else:
            
            self.players[index1].lose_die()
            #print(f"{self.players[index1].name} lost one die")
            
            if isinstance(self.players[index1], DQNAgentWrapper):
                self.current_reward += -(count - self.current_bid[0]) ** 2 - 25
            
            if isinstance(self.players[index2], DQNAgentWrapper):
                self.current_reward += 60
            
            if not self.players[index1].has_dice():
                #score_table.append(index1)
                #print(f"{self.players[index1].name} is eliminated")
                if not self.DQN_agent.has_dice():
                    self.done = True
                    self.score = self.set_score()
            
            if len([p for p in self.players if p.has_dice()]) == 1:
                if self.DQN_agent.has_dice():
                    self.current_reward += 350
                    self.score = 1
                    #print("Won the game ! reward:100")
                    
                self.done = True
                if self.score != 1:
                    self.score = self.set_score()
                
            return index1
        
    def start(self):
        
        self.set_new_round()
        self.new_round = False
        
        state = self.play_round()
        
        return state
        
        
    def step(self, action):
        
        proposal = action
        
        #print("DQN_Agent fait un pari:", proposal)
        
        self.current_index = self.agent_index
        
        current_player = self.players[self.current_index]
        
        if proposal == (0,0):
            if self.previous_index == None:
                
                raise ValueError(f"{current_player.name} said myth from start")
            else:
                starting_index = self.checking()
                self.new_round = True
                self.current_index = starting_index
                
        else:
            self.current_bid = proposal
            self.previous_index = self.current_index
            self.current_index = (self.previous_index + 1) % len(self.players)
            
       
        next_state = self.play_round()
        
        reward = self.current_reward
        self.current_reward = 0

        return next_state, reward, self.done