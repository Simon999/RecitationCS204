import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Define the grid world environment
class GridWorld:
    def __init__(self, width, height, goal, punish, obstacle, living_reward, noise):
        self.width = width
        self.height = height
        self.goal = goal
        self.punish = punish
        self.obstacle = obstacle
        self.living_reward = living_reward
        self.noise = noise
        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.actions = ['up', 'down', 'left', 'right']
        self.q_values = {s: {a: 0 for a in self.actions} for s in self.states}

    def is_terminal(self, state):
        return state == self.goal or state == self.punish

    def get_next_state(self, state, action):
        if self.is_terminal(state) or state == self.obstacle:
            return state

        # Define potential outcomes with noise
        if self.noise > 0:
            all_actions = self.actions[:]
            all_actions.remove(action)
            potential_actions = [action] * int((1 - self.noise) * 100) + all_actions
            action = random.choice(potential_actions)

        # Calculate next state based on action
        x, y = state
        if action == 'up':
            y = max(y - 1, 0)
        elif action == 'down':
            y = min(y + 1, self.height - 1)
        elif action == 'left':
            x = max(x - 1, 0)
        elif action == 'right':
            x = min(x + 1, self.width - 1)

        next_state = (x, y)

        # If next state is an obstacle, return the original state
        if next_state == self.obstacle:
            return state
        return next_state

    def get_reward(self, state, next_state):
        if next_state == self.goal:
            return 1
        elif next_state == self.punish:
            return -1
        else:
            return self.living_reward

    def update_q_values(self, alpha, gamma):
        """
        Update Q-values using the Q-learning update rule.
        """
        # Loop over all states
        for state in self.states:
            # Skip terminal and obstacle states
            if self.is_terminal(state) or state == self.obstacle:
                continue

            # Loop over all possible actions in each state
            for action in self.actions:
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(state, next_state)
                # Q-learning update rule
                best_future_q = max(self.q_values[next_state].values()) if not self.is_terminal(next_state) else 0
                self.q_values[state][action] += alpha * (reward + gamma * best_future_q - self.q_values[state][action])

    def draw(self, iteration):
        fig, ax = plt.subplots()
        cmap = ListedColormap(['white'])
        ax.imshow(np.ones((self.height, self.width)), cmap=cmap, extent=[0, self.width, 0, self.height])

        for state in self.states:
            x, y = state
            if state == self.obstacle:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color='grey', alpha=0.3))
                continue

            for action in self.actions:
                q_value = self.q_values[state][action]
                text_position = {
                    'up': (x + 0.5, y + 0.25),
                    'right': (x + 0.75, y + 0.5),
                    'down': (x + 0.5, y + 0.75),
                    'left': (x + 0.25, y + 0.5)
                }
                ax.text(*text_position[action], f"{q_value:.4f}", ha='center', va='center', fontsize=8)

        ax.add_patch(plt.Rectangle(self.goal, 1, 1, fill=True, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle(self.punish, 1, 1, fill=True, color='red', alpha=0.3))

        for x in range(self.width + 1):
            ax.axvline(x, color='black', linewidth=1)
        for y in range(self.height + 1):
            ax.axhline(y, color='black', linewidth=1)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.title(f"Q-values after {iteration} iterations")
        plt.gca().invert_yaxis()
        plt.show()

# Parameters for the grid world
width, height = 4, 3
goal = (3, 0)
punish = (3, 1)
obstacle = (1, 1)
living_reward = - 0.0
noise = 0.0

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 1  # discount factor
iterations = 100

# Initialize the grid world
grid_world = GridWorld(width, height, goal, punish, obstacle, living_reward, noise)

# Q-learning algorithm
for i in range(iterations):
    grid_world.update_q_values(alpha, gamma)
    grid_world.draw(i+1)