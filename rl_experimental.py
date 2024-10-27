# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:29:17 2024

@author: amude
"""

import numpy as np
import torch

from StockTradingEnv import StockTradingEnv
from RLAgent import RLAgent

def train_agent(agent, env, episodes):
    for episode in range(episodes):
        # Reset
        state = env.reset()
        hidden_state = (torch.zeros(1, 1, agent.model.hidden_size), torch.zeros(1, 1, agent.model.hidden_size))
        total_reward = 0
        
        while True:
            action, hidden_state = agent.choose_action(state, hidden_state)
            next_state, reward, done = env.step(action)
            
            # Train the agent
            agent.remember(state, action, reward, next_state, done)
            agent.replay(hidden_state)
            
            state = next_state # move to the next state
            total_reward += reward

            if done:
                break
        
        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

# Test agent after training
def test_agent(agent, env):
    state = env.reset()
    hidden_state = (torch.zeros(1, 1, agent.model.hidden_size), torch.zeros(1, 1, agent.model.hidden_size))
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)  # Predict action with stop-loss & take-profit
        next_state, reward, done = env.step(action)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f'Test result: Total Reward = {total_reward}')


if __name__ == '__main__':
    # Simulated historical stock prices (e.g., 100 days of random prices between $100 and $200)
    stock_data = np.random.uniform(100, 200, 100)

    # Initialize the environment
    env = StockTradingEnv(data=stock_data, initial_balance=10000)  # Starting with $10,000
    
    # Define RL model inputs
    input_size = 3  # [stock_price, stock_held, balance]
    hidden_size = 64
    output_size = 5  # [Buy/Sell/Hold, stop-loss %, take-profit %]

    # Initialize the RL agent
    agent = RLAgent(input_size=input_size, hidden_size=hidden_size, output_size=output_size, action_space=[0, 1, 2])
    
    # Train the agent
    print("Training the agent...")
    train_agent(agent, env, episodes=10)  # Train for 10 episodes
    
    # # Test the agent
    # print("\nTesting the agent...")
    # test_agent(agent, env)

