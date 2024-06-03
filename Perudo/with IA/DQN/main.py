# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:55 2024

@author: jules
"""


import numpy as np
from QNetwork import DQNAgent
from game import CustomGameEnv

def main():
    num_players = 4  # Adjust as necessary
    env = CustomGameEnv(num_players, None)  # Initialize environment without the agent first
    
    state_size = len(env.reset())
    action_size = len(env.generate_action_space((1, 0), num_players * 6))
    
    q_agent = DQNAgent(state_size, action_size)  # Initialize the DQN agent with the calculated sizes
    env.q_agent = q_agent  # Set the agent in the environment
    
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            
            env.start()
            
            if done:
                q_agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {q_agent.epsilon:.2}")
                break
            if len(q_agent.memory) > q_agent.batch_size:
                q_agent.replay()
    
        
        
if __name__ == "__main__":
    main()