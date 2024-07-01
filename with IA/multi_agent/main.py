# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:37:06 2024

@author: jules
"""

import torch
import time
from game import Game
from dqn_agent import DQNAgent
from player import DQNAgentWrapper
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_total_action_dim(num_players):
    
    total_initial_dice = num_players * 5
    total_action_dim = 1

    for a in range(1, total_initial_dice + 1):
            for b in range(1, 7):
                total_action_dim += 1
    return total_action_dim

def load_models(dqn_agents, num_agents):
    for i in range(num_agents):
        model_path = f"dqn_model_agent_{i+1}.pth"
        if os.path.exists(model_path):
            try:
                dqn_agents[i].load_model(model_path)
                print(f"Model for Agent {i+1} loaded successfully.")
            except FileNotFoundError:
                print(f"No model found for Agent {i+1}, starting fresh.")
        else:
            print(f"No model file found for Agent {i+1}, starting fresh.")

def mutate_agent(agent, mutation_rate=0.1):
    for param in agent.model.parameters():
        param.data += torch.randn_like(param) * mutation_rate

def select_best_agent(agents, mean_scores):
    # Sort agents based on their mean scores
    best_index = np.argmin(mean_scores)  # Find the index of the best score
    best_agent = agents[best_index]
    
    print(f"Top agent : {best_index + 1}")
    return best_index, best_agent

def replicate_and_mutate(agents, top_agent, mutation_rate=0.1):
    new_agents = []
    
    for _ in range(len(agents)):
        new_agent = DQNAgent(top_agent.state_dim, top_agent.action_dim)
        new_agent.model.load_state_dict(top_agent.model.state_dict())
        mutate_agent(new_agent, mutation_rate)
        new_agents.append(new_agent)
    return new_agents

def evaluate_agents(agents, episodes):
    
    game = Game(num_players, agents)
    
    load_models(agents, num_players)
    
    final_scores = []
    historic_of_rewards = []
    for episode in range(episodes):
        game.reset()
        e = game.players[game.current_index].dqn_agent.epsilon
        
        for p in game.players[1:]:
            p.dqn_agent.epsilon_decay = 0.0
        
        done = False
        previous_state = None
        previous_action = None
        previous_player = None
        total_rewards = [0]*(num_players)

        
        while not done:
            
            time.sleep(1)

            current_player = game.players[game.current_index]
            
            if current_player.has_dice():

                if isinstance(current_player, DQNAgentWrapper):
                    state = current_player.get_state(game.current_bid, game.total_dice, num_players)
                    action = current_player.dqn_agent.choose_action(state, num_players)
                
                else:
                    action = current_player.brusque_bet(game.current_bid, game.total_dice)
                
                print(f"{current_player.name} calls {action}")
                
                if action[0] == 0 and action[1] != 0:
                    raise ValueError(f"current_bid : {game.current_bid}")
    
                next_state, reward, done = game.step(action)
    
                if previous_state is not None and previous_action is not None and isinstance(previous_player, DQNAgentWrapper):
                    
                    print(f"{previous_player.name}, {previous_action}, reward: {reward}")
                    previous_player.dqn_agent.replay_buffer.push(previous_state, previous_action, reward, state, done)
                    if len(previous_player.dqn_agent.replay_buffer) > previous_player.dqn_agent.batch_size:
                        previous_player.dqn_agent.learn()
                
                if isinstance(current_player, DQNAgentWrapper):
                    if previous_player is not None:
                        total_rewards[game.players.index(previous_player)] += reward
                    previous_state = state
                    previous_action = current_player.dqn_agent.encode_action(action)
                    previous_player = current_player
                else:
                    previous_player = current_player
            
            else:
                
                game.current_index = (game.current_index + 1) % game.num_players
        
        if done:
            for p in game.players:
                if isinstance(p, DQNAgentWrapper):
                    p.dqn_agent.update_target_model()
            print(f"episode: {episode}/{episodes}, e: {e:.2f}")

        final_scores.append(game.score_table)
        historic_of_rewards.append(total_rewards)
                
    
    ep = np.arange(episodes)
    # Plot all curves on the same graph
    plt.figure(figsize=(10, 6))

    for i in range(num_players):
        y_values = [row[i] for row in historic_of_rewards]
        plt.plot(ep, y_values, label=f'Agent {i+1}')

    plt.xlabel('e')
    plt.ylabel('x')
    plt.title('total_reward of Agents')
    plt.legend()
    plt.grid(True)
    
    plt.show()
          
    mean_scores = [np.mean(values) for values in zip(*final_scores)]
    
    name = np.arange(1, num_players + 1)
    plt.bar(name, mean_scores)
    plt.xlabel('Agent name')
    plt.ylabel('Mean score')
    plt.title('Mean score of Agents')
    plt.legend()
    
    plt.show()
    
    return mean_scores

def train_dqn_agent(episodes, generations, num_players, mutation_rate=0.1):
    state_dim = 8 # 6 (dice) + 1 (encoded current bid) + 1 (total dice))
    action_dim = generate_total_action_dim(num_players)

    agents = [DQNAgent(state_dim, action_dim) for _ in range(num_players)]
    
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations}")

        # Evaluate agents
        scores = evaluate_agents(agents, episodes)

        # Select the best agents
        best_index, top_agent = select_best_agent(agents, scores)

        # Replicate and mutate agents
        agents = replicate_and_mutate(agents, top_agent, mutation_rate)

        # Save the top agents
        
        top_agent.save_model(f"dqn_model_agent_{best_index + 1}_gen_{generation}.pth")

    print("Training completed.")
    

    
if __name__ == "__main__":
    episodes = 1
    generations = 1
    num_players = 6
    train_dqn_agent(episodes, generations, num_players)
