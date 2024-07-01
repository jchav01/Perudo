# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:55 2024

@author: jules
"""

import time
from game import Game
import matplotlib.pyplot as plt
from Qlearning import QLearningAgent, QLearningAgentWrapper
from performances import PerformanceTracker

def main():
    num_players = 5
    q_agent = QLearningAgent()
    
    try:
        q_agent.load_q_table('q_table.pkl')
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("No Q-table found, starting fresh.")
    
    time.sleep(3)
        
    results = [0, 0, 0, 0, 0]  # Add a spot for Q-Agent
    strategies = ['shy', 'prudent', 'brusque', 'bluffeur', 'Q-Agent']
    counter = []
    
    performance_tracker = PerformanceTracker()
    
    for n in range(1, 2):
        counter.append(n)
        game = Game(num_players, q_agent = q_agent)
        q_agent.action_space = game.generate_action_space((1, 0), game.count_total_dice())
        
        if n%100==0:
            print(n)
        
        score_table, total_reward = game.start()
        
        score_table.reverse()
        
        winner_index = score_table[0]
        
        winner = game.players[winner_index]
        
        if winner.strategy == 'shy':
            results[0] += 1
        elif winner.strategy == 'prudent':
            results[1] += 1
        elif winner.strategy == 'brusque':
            results[2] += 1
        elif winner.strategy == 'bluffeur':
            results[3] += 1
        elif winner.strategy == 'q_learning':
            results[4] += 1
        
        # Track the position of the Q-learning agent
        for idx in score_table:
            if isinstance(game.players[idx], QLearningAgentWrapper):
                agent_position = idx
                performance_tracker.record_position(agent_position)
        
        performance_tracker.record_reward(total_reward)
        
        print(f"\n winner : {game.players[winner_index].name}\n")
    
    q_agent.save_q_table('q_table.pkl')
    print("Q-table saved successfully.")
        
    plot_results(results, strategies)
    performance_tracker.plot_positions(num_players)
    performance_tracker.plot_rewards()
    

def plot_results(results, strategies):
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, results, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Strategies')
    plt.ylabel('Number of Wins')
    plt.title('Number of Wins per Strategy')
    plt.show()
    
if __name__ == "__main__":
    main()