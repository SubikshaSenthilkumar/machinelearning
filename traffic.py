import numpy as np
import random
import matplotlib.pyplot as plt

class TrafficSignalControl:
    def __init__(self, num_directions=4):
        self.num_directions = num_directions
        self.max_traffic_per_direction = 10  # Maximum number of vehicles to be represented
        self.q_table = np.zeros((self.max_traffic_per_direction, self.max_traffic_per_direction, self.max_traffic_per_direction, self.max_traffic_per_direction, num_directions))  # Q-table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99

    def get_state(self, traffic_counts):
        # Map traffic counts to state space (0-9 vehicles per direction)
        state = tuple(min(count, self.max_traffic_per_direction - 1) for count in traffic_counts)
        return state

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randint(0, self.num_directions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        # Ensure state is in tuple format for indexing
        max_future_q = np.max(self.q_table[next_state])  # Q-value of the next state
        current_q = self.q_table[state + (action,)]  # Accessing Q-value using state and action
        # Update the Q-value using the Q-learning formula
        self.q_table[state + (action,)] = (1 - self.learning_rate) * current_q + \
                                            self.learning_rate * (reward + self.discount_factor * max_future_q)

    def decay_exploration(self):
        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay

    def simulate_traffic(self):
        # Simulate traffic counts (0-9 vehicles in each direction)
        return np.random.randint(0, 10, self.num_directions)

# Simulation parameters
num_episodes = 10000
traffic_signal = TrafficSignalControl(num_directions=4)
rewards = []

for episode in range(num_episodes):
    traffic_counts = traffic_signal.simulate_traffic()
    state = traffic_signal.get_state(traffic_counts)
    action = traffic_signal.choose_action(state)
    
    # Simulate reward: assume each vehicle waiting adds to total waiting time
    reward = -np.sum(traffic_counts)  # Negative waiting time (to minimize)
    rewards.append(reward)

    # Update Q-values
    next_traffic_counts = traffic_signal.simulate_traffic()  # Next state traffic counts
    next_state = traffic_signal.get_state(next_traffic_counts)
    traffic_signal.update_q_value(state, action, reward, next_state)

    # Decay exploration rate
    traffic_signal.decay_exploration()

# Plot total rewards over episodes
plt.plot(rewards)
plt.title('Traffic Signal Control System Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()
