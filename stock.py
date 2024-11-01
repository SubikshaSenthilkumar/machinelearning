import gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Create a mock environment for stock trading
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(3)  # Actions: 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float16)  # Price history
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.stock_held = 0
        self.current_step = 0
        self.net_worth = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.stock_held,
        ]) / 1000  # Normalized
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        
        # Take action
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.stock_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.stock_held > 0:
                self.stock_held -= 1
                self.balance += current_price
        
        self.current_step += 1
        self.net_worth = self.balance + self.stock_held * current_price
        
        # Done if at the end
        done = self.current_step >= len(self.df) - 1
        reward = self.net_worth - self.initial_balance
        obs = self._next_observation()
        return obs, reward, done, {}

# Load data (replace with your data source)
data = pd.DataFrame({
    'Open': np.random.rand(1000) * 1000,
    'High': np.random.rand(1000) * 1000,
    'Low': np.random.rand(1000) * 1000,
    'Close': np.random.rand(1000) * 1000,
})

# Create environment
env = StockTradingEnv(data)

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.95      # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.1

# Discretize the continuous state into integers
def discretize_state(state):
    return tuple((state * 10).astype(int))  # Scale and convert to integer for indexing

# Initialize Q-table
q_table = {}

# Tracking performance metrics
rewards_per_episode = []

# Training
num_episodes = 1000
for episode in range(num_episodes):
    state = discretize_state(env.reset())
    done = False
    total_reward = 0
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)
            action = np.argmax(q_table[state])  # Exploit learned values
            
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-table
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(env.action_space.n)
        
        q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        
        state = next_state
        total_reward += reward  # Track total reward

    rewards_per_episode.append(total_reward)  # Log episode reward
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon

print("Training complete.")

# Plot total rewards over episodes
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.show()
