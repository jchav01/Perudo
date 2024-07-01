# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:55 2024

@author: jules
"""

from QNetwork import DQNAgent, DQNAgentWrapper
from game import CustomGameEnv
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_total_action_space(num_players):
    
    total_initial_dice = num_players * 5
    total_action_space = []
    total_action_space.append((0,0))
    for a in range(1, total_initial_dice + 1):
            for b in range(1, 7):
                total_action_space.append((a, b))
    return total_action_space

def main():
    num_players = 6
    
    state_size = 8
    max_actions = len(generate_total_action_space(num_players))
    
    dqn_agent = DQNAgent(state_size, max_actions)  # Initialize the DQN agent with the calculated sizes
    dqn_agent_wrapper = DQNAgentWrapper("DQNAgent", dqn_agent)
    
    try:
        dqn_agent.load_model("dqn_model.pth")
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("No Q-table found, starting fresh.")
        
    time.sleep(1)
    env = CustomGameEnv(num_players, dqn_agent_wrapper)
    
    episodes = 800 # Adjust the number of episodes as necessary
    score_table = []

    for e in range(episodes):
        
        env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        state = env.start()

        while True:
            
            action = dqn_agent.choose_action(state, num_players)
            next_state, reward, done = env.step(action)
            dqn_agent.store_transition(state, action, reward, next_state, done)
            total_reward += reward
            
            if len(dqn_agent.replay_buffer) > dqn_agent.batch_size:
                dqn_agent.learn()
            
            state = next_state
            
            if done:
                dqn_agent.update_target_model()
                print(f"episode: {e}/{episodes}, reward: {total_reward}, score: {env.score}/{num_players} e: {dqn_agent.epsilon:.2f}")
                break
        
        score_table.append(env.score)
    
    dqn_agent.save_model("dqn_model.pth")
    print("Modèle sauvegardé.")

    all_positions = np.arange(1, num_players + 1)
    positions, counts = np.unique(score_table, return_counts=True)
    
    all_counts = np.zeros_like(all_positions)
    for pos, count in zip(positions, counts):
        all_counts[pos - 1] = count

    plt.bar(all_positions, all_counts)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('DQN Agent Position Distribution')
    plt.xticks(all_positions)
    plt.show()

 
if __name__ == "__main__":
    main()