# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:55 2024

@author: jules
"""

from game import Game
import matplotlib.pyplot as plt



def main():
    # Initialisation du jeu avec un nombre fixe de joueurs par exemple.
    num_players = 8
    
    results = [0, 0, 0, 0]
    strategies = ['shy', 'prudent', 'brusque', 'bluffeur']
    
    for n in range(1, 1001):
        game = Game(num_players)
        winner = game.start()
        if winner.strategy == 'shy':
            results[0] += 1
        elif winner.strategy == 'prudent':
            results[1] += 1
        elif winner.strategy == 'brusque':
            results[2] += 1
        elif winner.strategy == 'bluffeur':
            results[3] += 1

    # Plotting the results
    plot_results(results, strategies)

def plot_results(results, strategies):
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, results, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Strategies')
    plt.ylabel('Number of Wins')
    plt.title('Number of Wins per Strategy')
    plt.show()

if __name__ == "__main__":
    main()
