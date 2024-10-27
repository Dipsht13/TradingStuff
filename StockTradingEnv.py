# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:27:12 2024

@author: amude
"""

import pandas as pd


class StockTradingEnv:
    def __init__(self, data, per_minute_data, initial_balance):
        self.data = data  # Historical stock data (prices, volume, etc.)
        self.per_minute_data = per_minute_data
        self.initial_balance = initial_balance
        self.reset()
    
    
    def reset(self):
        # Reset the environment to its initial state
        self.current_date = self.data.iloc[0]['Date']
        self.current_ix = 0
        self.balance = self.initial_balance
        self.stock_held = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.earnings = 0
        
        return self.get_state()
    
    
    def get_state(self):
        # Return the current market state (e.g., price, stock held, balance)
        stock_price = self.data.iloc[self.current_ix]['Close']
        return [stock_price, self.stock_held, self.balance]
    
    
    def step(self, action):
        # Unpack action (e.g., [Buy/Sell/Hold, Stop-loss %, Take-profit %])
        trade_action = action[0]  # Buy or Hold
        stop_loss_pct = action[1]  # Stop-loss percentage
        take_profit_pct = action[2]  # Take-profit percentage
        num_shares = action[3]  # Number of shares to buy/sell

        stock_price = self.data.iloc[self.current_ix]['Close']
        
        # Execute trade actions
        if trade_action == 1:  # Buy
            total_cost = num_shares * stock_price  # Determine balance required
            if self.balance >= total_cost:  # Ensure enough balance
                self.stock_held += num_shares
                self.balance -= total_cost
                self.entry_price = stock_price  # Set the entry price for the trade
                self.stop_loss = stock_price * (1 - stop_loss_pct)  # Calculate stop-loss price
                self.take_profit = stock_price * (1 + take_profit_pct)  # Calculate take-profit price
                
                self.propagate_trade()

        elif trade_action == 2:  # Sell
            if self.stock_held > 0:  # Ensure there are stocks to sell
                self.stock_held -= 1
                self.balance += stock_price
                reward = stock_price - self.entry_price  # Calculate profit/loss for this trade
                self.entry_price = None  # Reset the entry price after selling
                self.stop_loss = None
                self.take_profit = None
        
        # Check stop-loss and take-profit conditions
        if self.entry_price is not None:
            if stock_price <= self.stop_loss:
                # Stop-loss triggered, sell the stock
                self.stock_held -= 1
                self.balance += stock_price
                reward = self.stop_loss - self.entry_price  # Loss incurred
                self.entry_price = None  # Reset trade after stop-loss is hit
                self.stop_loss = None
                self.take_profit = None
            
            elif stock_price >= self.take_profit:
                # Take-profit triggered, sell the stock
                self.stock_held -= 1
                self.balance += stock_price
                reward = self.take_profit - self.entry_price  # Profit earned
                self.entry_price = None  # Reset trade after take-profit is hit
                self.stop_loss = None
                self.take_profit = None
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.stock_held * stock_price)
        
        # Move to the next time step
        self.current_step += 1
        
        done = self.current_step >= len(self.data['price']) - 1  # Check if episode ends
        
        return self.get_state(), reward, done  # Return new state, reward, and done flag
    
    
    def propagate_trade(self):
        # Apply stop loss/take profit after buying
        # Going to use the 'Close' from the per-minute data to determine running costs       
        
        pm_ix = self.per_minute_data.loc[self.per_minute_data['Date'].dt.date == self.current_date].index[0]
        while True:
            current_price = self.per_minute_data.iloc[pm_ix]['Close']
            
            if self.stop_loss >= current_price:
                amount_lost = (self.entry_price - self.stop_loss) * self.stock_held
                
                self.earnings -= amount_lost
                self.balance += self.stop_loss * self.stock_held
                
                self.entry_point = None
                self.stop_loss = None
                self.take_profit = None
                self.stock_held = None
                
                self.current_date = self.per_minute_data.iloc[pm_ix]['Date'].date
                self.current_ix = self.data.loc[self.data['Date'].dt.date == self.current_date].index
                
                break
            
            elif self.take_profit <= current_price:
                amount_won = (self.take_profit - self.entry_point) * self.stock_held
                
                self.earnings += amount_won
                self.balance += self.take_profit * self.stock_held
                
                self.entry_point = None
                self.stop_loss = None
                self.take_profit = None
                self.stock_held = None
                
                self.current_date = self.per_minute_data.iloc[pm_ix]['Date'].date
                self.current_ix = self.data.loc[self.data['Date'].dt.date == self.current_date].index
                                
                break
            
            else:
                pm_ix += 1
            
        return
        
        
