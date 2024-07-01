# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:07:47 2024

@author: jules
"""
import random
from player import Player, DQNAgentWrapper

class Game:
    def __init__(self, num_players, dqn_agents):
        self.num_players = num_players
        self.players = self.create_players(num_players, dqn_agents)
        self.current_bid = (1, 0)
        self.score_table = [-1] * num_players
        self.total_dice = num_players * 5
        self.current_index = 0
        self.previous_index = None
        self.winner_index = None

        

    def create_players(self, num_players, dqn_agents):
        players = [DQNAgentWrapper(f"Agent {i+1}", dqn_agents[i]) for i in range(num_players)]

        return players

    def reset(self):
        for player in self.players:
            player.dice = [random.randint(1, 6) for _ in range(5)]
            player.eliminated = False
        self.current_bid = (1, 0)
        self.total_dice = self.num_players * 5
        self.score_table = [-1] * self.num_players
        self.current_index = 0
        self.previous_index = None
        self.show_dice()

    def count_total_dice(self):
        return sum(len(player.get_dice()) for player in self.players if player.has_dice())
    
    def show_dice(self):
        for p in self.players:
            print(f"{p.name} : {p.dice}")

    def set_new_round(self, max_history_length=10):
        for p in self.players:
            p.roll_dice()
        
        self.show_dice()
        self.current_bid = (1,0)
        self.previous_index = None
        
    def reset_history_bid(self, max_history_length=10):
        for i in range(max_history_length):
            self.bid_history[i] = ((-1, -1))
    
    def get_state(self):
        state = []
        for player in self.players:
            state.extend(player.dice)
        state.append(self.current_bid)
        state.append(self.total_dice)
        state.extend(self.flatten_bid_history())
        return state

    def flatten_bid_history(self, max_history_length=10):
        flattened_history = []
        for bid in self.bid_history[-max_history_length:]:
            flattened_history.extend(bid)
        while len(flattened_history) < max_history_length * 2:
            flattened_history.extend((-1, -1))
        return flattened_history
    
    def set_score(self):
        count = 0
        for p in self.players:
            if p.has_dice():
                count += 1
        return count + 1

    def step(self, action):
        
        done = False
        reward = 0
        
        i = self.current_index
        
        current_player = self.players[i]
        
        
        if action == (0,0):
            if self.previous_index == None:
                raise ValueError(f"{current_player.name} said myth from start")
            else:
                loser_index, self.winner_index, reward = self.checking()
                self.current_index = loser_index
                self.set_new_round()          

        else:
            if self.winner_index != None:
                reward +=  self.players[self.winner_index].reward
                self.players[self.winner_index].reward = 0
                self.winner_index = None
            reward += 5
            self.previous_index = i
            i = (i + 1) % self.num_players
            self.current_index = i
            current_player = self.players[i]
            self.current_bid = action
        
        
        
        if len([p for p in self.players if p.has_dice()]) == 1:
            done = True
            
        
        
        return current_player.get_state(self.current_bid, self.total_dice, self.num_players), reward, done


    def checking(self):
        
        index1 = self.previous_index
        index2 = self.current_index

        
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
            reward = (count - self.current_bid[0])**2 + 10
            self.players[index2].reward =  -(count - self.current_bid[0])**2 - 10
            
            if not self.players[index2].has_dice():
                print(f"{self.players[index2].name} is eliminated")
                if len([p for p in self.players if p.has_dice()]) != 1:
                    self.score_table[index2] = self.set_score()
                else:
                    self.score_table[index2] = 2
                    self.score_table[index1] = 1
                    
            self.total_dice -= 1
            
            
            return index2, index1, reward
        
        else:
            
            self.players[index1].lose_die()
            print(f"{self.players[index1].name} lost one die")
            reward = - (count - self.current_bid[0])**2 - 10
            self.players[index2].reward = (count - self.current_bid[0])**2 + 10
            
            
            if not self.players[index1].has_dice():
                print(f"{self.players[index1].name} is eliminated")
                if len([p for p in self.players if p.has_dice()]) != 1:
                    self.score_table[index1] = self.set_score()
                else:
                    self.score_table[index1] = 2
                    self.score_table[index2] = 1
                
            self.total_dice -= 1

                
            return index1, index2, reward
        
                    
    
