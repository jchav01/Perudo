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
        self.current_bid = None
        self.state = self.reset()
        self.total_dice = num_players * 5
        self.done = False

        
        
    def reset(self):
        self.done = False
        self.current_player_index = 0
        organized_dice = [(0, n+1) for n in range(6)]
        total_dice = self.num_players * 5
        starting_bid = list((1, 0))
        flattened_dice = [item for sublist in organized_dice for item in sublist]
        self.state = np.array(flattened_dice + starting_bid + [total_dice])
        
        self.players = self.create_players(self.num_players, self.DQN_agent)
        
        return self.state

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
    
    def get_agent_state(self):
        current_bid = self.current_bid
        total_dice = self.total_dice
        dice = self.DQN_agent.get_dice()
        dice_state = [0]*6
        for dice in dice:
            dice_state[dice-1] += 1
        encoded_bid = current_bid[0]*10 + current_bid[1]
        return np.array(list(dice_state) + [encoded_bid] + [total_dice])

    def show_players_dice(self):
        print("\n Current dice of all players:")
        for player in self.players:
            if player.has_dice():
                print(f"{player.name}: {player.get_dice()} {player.organize_dice()}")

    def count_total_dice(self):
        return sum(len(player.get_dice()) for player in self.players if player.has_dice())

    def count_active_players(self):
        return sum(1 for player in self.players if player.has_dice())
    
    def get_total_dice(self, state):
        total_dice = state[-1]
        return total_dice
    
    def get_current_bid(self, state):
        current_bid = []
        current_bid.append(state[-3])
        current_bid.append(state[-2])
        return current_bid
    
    def start(self):
        
        starting_index = 0
        
        reward = 0

        done = False
        
        new_round = True
        
        for player in self.players:
            if player.has_dice():
                player.roll_dice()
                
        self.show_players_dice()
    
        self.current_bid = (1,0)
        previous_index = None

        while(new_round == True):   
            
            new_round = False
            previous_index = None
            
            new_round, state, previous_index, reward, done = self.play_round(new_round, starting_index, previous_index, reward, done)
        
        print("OUT1", new_round)
        
        return new_round, state, previous_index, reward, done
    
    def play_round(self, new_round, starting_index, previous_index, reward, done):
        
        total_dice = self.total_dice
        
        i = starting_index
        
        current_player = self.players[i]
        
        while not isinstance(current_player, DQNAgentWrapper):
                
            if current_player.has_dice():
                
                proposal = current_player.bet(self.current_bid, total_dice)
                    
                if proposal != (0, 0):
                    self.current_bid = proposal
                    if previous_index == self.players.index(self.DQN_agent):
                        
                        reward = 5
                 
                else:
                    if previous_index == None:
                        raise ValueError(f"{current_player.name} said myth from start")
                    else:
                        starting_index, reward, done = self.checking(i, previous_index, done)
                        new_round = True
                        self.current_bid = (1,0)
                        break
                
                previous_index = i
                
                print(f"{current_player.name} ({current_player.strategy})  fait un pari: {self.current_bid}")
                
    
            i = (i + 1) % len(self.players)
            current_player = self.players[i]
            
        state = self.get_agent_state()
        
        print("OUT2", new_round)
        return new_round, state, previous_index, reward, done
        
        
    def checking(self, current_index, previous_index, done):
        
        index1 = previous_index
        index2 = current_index
        
        current_player = self.players[index2]
        
        reward = 0
        count = 0
        for k in range(len(self.players)):
            for l in range(len(self.players[k].dice)):
                if self.players[k].dice[l] == self.current_bid[1]:
                    count += 1
            
        print(f"{self.players[index2].name} challenges {self.players[index1].name}: {self.current_bid}")
        print(f"result: ({count}, {self.current_bid[1]})") 
        
        if count >= self.current_bid[0]:
            
            self.players[index2].lose_die()
            print(f"{self.players[index2].name} lost one die")
            
            if isinstance(current_player, DQNAgentWrapper):
                reward = -(count - self.current_bid[0]) ** 2
                
            if isinstance(self.players[index2], DQNAgentWrapper):
                reward = 20
         
            if not self.players[index2].has_dice():
                #score_table.append(index2)
                print(f"{self.players[index2].name} is eliminated")
                if not self.DQN_agent.has_dice():
                    done = True
            
            self.total_dice -= 1
            
            if len([p for p in self.players if p.has_dice()]) == 1:
                if self.DQN_agent.has_dice():
                    reward = 300
                    print("Won the game ! reward:", reward)
                
                done = True
            
            return index2, reward, done
        
        else:
            
            self.players[index1].lose_die()
            print(f"{self.players[index1].name} lost one die")
            
            if isinstance(self.players[index1], DQNAgentWrapper):
                reward = -(count - self.current_bid[0]) ** 2 
            
            if isinstance(self.players[index2], DQNAgentWrapper):
                reward = 20
            
            if not self.players[index1].has_dice():
                #score_table.append(index1)
                print(f"{self.players[index1].name} is eliminated")
                if not self.DQN_agent.has_dice():
                    done = True
            
            self.total_dice -= 1
            
            if len([p for p in self.players if p.has_dice()]) == 1:
                if self.DQN_agent.has_dice():
                    reward = 300
                    print("Won the game ! reward:", reward)
                    
                done = True
                
            return index1, reward, done
        
    def step(self, new_round, action, previous_index, done):
        
        if new_round == True:
            for player in self.players:
                if player.has_dice():
                    player.roll_dice()
                    
            self.show_players_dice()
        
            self.current_bid = (1,0)
            previous_index = None
        reward = 0
        
        proposal = action
        
        print("DQN_Agent fait un pari:", proposal)
        
        current_player = self.DQN_agent
        
        i = self.players.index(current_player)
        
        if proposal == (0,0):
            if previous_index == None:
                print(self.get_agent_state())
                print(self.current_bid)
                
                raise ValueError(f"{current_player.name} said myth from start")
            else:
                reward, starting_index, done = self.checking(i, previous_index, done)
            
            new_round = True
            
        else:
            self.current_bid = proposal
            previous_index = i
            i = (i + 1) % len(self.players)
            starting_index = i
        
            
        new_round, next_state, previous_index, reward, done = self.play_round(new_round, starting_index, previous_index, reward, done)
         
        print("OUT3", new_round)
        return new_round, next_state, previous_index, reward, done