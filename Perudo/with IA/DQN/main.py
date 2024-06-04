# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:21:55 2024

@author: jules
"""



from QNetwork import DQNAgent, DQNAgentWrapper
from game import CustomGameEnv
#from performances import PerformanceTracker
import matplotlib.pyplot as plt


def main():
    num_players = 5
    
    state_size = 8
    action_size = 2
    
    dqn_agent = DQNAgent(state_size, action_size)  # Initialize the DQN agent with the calculated sizes
    dqn_agent_wrapper = DQNAgentWrapper("DQNAgent", dqn_agent)
    
    env = CustomGameEnv(num_players, dqn_agent_wrapper)
    
    """
    results = [0, 0, 0, 0, 0]
    strategies = ['shy', 'prudent', 'brusque', 'bluffeur', 'Q-Agent']
    
    performance_tracker = PerformanceTracker()
    """
    episodes = 10  # Adjust the number of episodes as necessary

    for e in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        done = False
        
        new_round, state, previous_index, reward, done = env.start()
        
        
        while True:
            
            action = dqn_agent.choose_action(state)
            new_round, next_state, previous_index, reward, done = env.step(new_round, action, previous_index, done)
            print("reward: ", reward)
            dqn_agent.replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            
            if len(dqn_agent.replay_buffer) > dqn_agent.batch_size:
                dqn_agent.replay()
            
            state = next_state
            
            if done:
                dqn_agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {total_reward}, e: {dqn_agent.epsilon:.2f}")
                break
        
        dqn_agent.update_epsilon()
        
        """
        score_table, _, _ = env.get_score_table()
        score_table.reverse()
        
        winner_index = score_table[0]
        winner = env.players[winner_index]
        
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
        
        for idx in score_table:
            if isinstance(env.players[idx], DQNAgentWrapper):
                agent_position = idx
                performance_tracker.record_position(agent_position)
        
        performance_tracker.record_reward(total_reward)
    
    plot_results(results, strategies)
    performance_tracker.plot_positions(num_players)
    performance_tracker.plot_rewards()
    """

def plot_results(results, strategies):
    plt.figure(figsize=(10, 6))
    plt.bar(strategies, results, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Strategies')
    plt.ylabel('Number of Wins')
    plt.title('Number of Wins per Strategy')
    plt.show()
            
        
if __name__ == "__main__":
    main()