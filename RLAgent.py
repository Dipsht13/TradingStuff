# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:29:52 2024

@author: amude
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 1. Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.fc(out[:, -1, :])  # Only take the output from the last time step
        return out, hidden_state

# 2. Define the RL Agent with LSTM
class RLAgent:
    def __init__(self, input_size, hidden_size, output_size, action_space, buffer_size=1000, batch_size=32):
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Experience replay buffer
        self.memory = deque(maxlen=self.buffer_size)

        # LSTM model and optimizer
        self.model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def choose_action(self, state, hidden_state):
        # Epsilon-greedy strategy for exploration/exploitation
        if np.random.rand() <= self.epsilon:
            trade_action = random.choice([0, 1, 2])  # Random action
            stop_loss_pct = random.uniform(0.01, 0.1)  # Random stop-loss
            take_profit_pct = random.uniform(0.01, 0.2)  # Random take-profit
            num_shares = random.randint(1, 10)  # Random number of shares to buy/sell
            return [trade_action, stop_loss_pct, take_profit_pct, num_shares], hidden_state

        # Convert state to torch tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)

        # Forward pass through the LSTM model
        with torch.no_grad():
            q_values, hidden_state = self.model(state, hidden_state)
            q_values = q_values.squeeze(0)  # Remove batch dimension

        # Action: [Buy/Sell/Hold, Stop-loss %, Take-profit %, number of shares]
        action = torch.argmax(q_values[:3]).item()  # Get Buy/Sell/Hold action (first 3 outputs)
        stop_loss_pct = q_values[3].item()  # Stop-loss percentage (4th output)
        take_profit_pct = q_values[4].item()  # Take-profit percentage (5th output)
        num_shares = int(q_values[5].item())  # Number of shares (6th output)

        return [action, stop_loss_pct, take_profit_pct, num_shares], hidden_state
    

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, hidden_state):
        # If not enough samples in memory, return
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from the memory
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).unsqueeze(1)  # (batch_size, 1, input_size)
        next_state_batch = torch.FloatTensor(next_state_batch).unsqueeze(1)  # (batch_size, 1, input_size)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)

        # Forward pass for current states and next states
        q_values, _ = self.model(state_batch, hidden_state)
        next_q_values, _ = self.model(next_state_batch, hidden_state)

        # Target for Q-learning
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * torch.max(next_q_values, dim=1, keepdim=True)[0]

        # Loss function: Mean Squared Error between target and actual Q-values
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay for exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

