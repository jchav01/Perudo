# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:28 2024

@author: jules
"""

from player import Player


class Game:
    def __init__(self, num_players):
        self.players = self.create_players(num_players)
        self.current_bid = None
        self.current_player_index = 0
    

    def create_players(self, num_players):
        players = []
        strategies = ['shy','prudent', 'brusque', 'bluffeur']
        for i in range(num_players):
            strategy = strategies[i % len(strategies)]
            players.append(Player(f"Player {i+1}", strategy))
        return players
    
    
    
    def start(self):
        loser_indice = 0
        winner_indice = 0
        while len([p for p in self.players if p.has_dice()]) > 1: 
            loser_indice, winner_indice = self.play_round(loser_indice)
        
        winner = self.players[winner_indice]
        print("\n \n Winner:", winner.name)
        
        return self.players[winner_indice]
    
    def show_players_dice(self):
        print("Current dice of all players:")
        for player in self.players:
            if player.has_dice():
                print(f"{player.name}: {player.get_dice()} ({player.organize_dice()}")
    
    def count_total_dice(self):
        total_dice = sum(len(player.get_dice()) for player in self.players if player.has_dice())
        return total_dice
    
    def count_active_players(self):
        count = sum(1 for player in self.players if player.has_dice())
        return count
    
    def play_round(self, indice):
        
        for player in self.players:
            if player.has_dice():
                player.roll_dice()
        
        
        proposal = (1, 0)
        self.current_bid = proposal
        
        total_dice = self.count_total_dice()
        previous_indice = None  
        i = indice
        
        while True:
            current_player = self.players[i]
            
            
            if current_player.has_dice():
                
                proposal = current_player.bet(self.current_bid, total_dice)
                
                if proposal != (0,0):
                    self.current_bid = proposal
                else:
                    break
                
                previous_indice = i
            
            i = (i+ 1) % len(self.players)
            
        
        indice1 = previous_indice

        indice2 = self.players.index(current_player)
        
        print(f"{self.players[indice2].name} challenges {self.players[indice1].name}: {self.current_bid}")

        count = 0
        
        for k in range(len(self.players)):
            for l in range(len(self.players[k].dice)):
                if self.players[k].dice[l] == self.current_bid[1]:
                    count += 1
                    
        print(f"result : ({count}, {self.current_bid[1]})")
        
        if count >= self.current_bid[0]:
            self.players[indice2].lose_die()
            print(f"{self.players[indice2].name} lost one die")
            
            if not self.players[indice2].has_dice():
                print(f"{self.players[indice2].name} is eliminated")
            
            return indice2, indice1

        else:
            self.players[indice1].lose_die()
            print(f"{self.players[indice1].name} lost one die")
            
            if not self.players[indice1].has_dice():
                print(f"{self.players[indice1].name} is eliminated")
        
            return indice1, indice2
            
        