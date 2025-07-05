import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your environment class (ensure this file is in the same directory)
from env.env_SingleAgent import SimpleSingleAgentEnv

# Q-Learning hyperparameters (can be modified for experiments)
ALPHAS = [0.1, 0.5]          # Learning rates to experiment
GAMMAS = [0.9, 0.99]         # Discount factors to experiment
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_DECAY = 0.995        # Exploration decay per episode
MIN_EPSILON = 0.01
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200
N_ACTIONS = 4                # Up, Down, Left, Right


def state_to_index(env):
    x, y = env.agent_pos
    has_item = int(env.agent_has_item)
    return (x, y, has_item)


class TabularQLearner:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = N_ACTIONS
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def train(self):
        rewards_per_episode = []
        steps_per_episode = []
        success_per_episode = []
        epsilon = EPSILON_START

        for ep in range(1, NUM_EPISODES + 1):
            self.env.reset()
            state = state_to_index(self.env)
            total_reward = 0
            steps = 0
            done = False
            success = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = self.choose_action(state, epsilon)
                _, reward, done, _ = self.env.step(action)
                next_state = state_to_index(self.env)

                # Determine if this done is a successful rescue
                if done and reward >= 10:
                    success = True

                old_val = self.Q[state][action]
                target = reward + (0 if done else self.gamma * np.max(self.Q[next_state]))
                self.Q[state][action] = old_val + self.alpha * (target - old_val)

                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
            success_per_episode.append(1 if success else 0)

            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        return rewards_per_episode, steps_per_episode, success_per_episode

    def evaluate(self, episodes=100):
        total_rewards = []
        for _ in range(episodes):
            self.env.reset()
            state = state_to_index(self.env)
            ep_reward = 0
            done = False
            while not done:
                action = int(np.argmax(self.Q[state]))
                _, reward, done, _ = self.env.step(action)
                ep_reward += reward
                state = state_to_index(self.env)
            total_rewards.append(ep_reward)
        return np.mean(total_rewards)


if __name__ == "__main__":
    # Experiment over hyperparameters
    results = {}
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            env = SimpleSingleAgentEnv(size=5, randomize=False)
            agent = TabularQLearner(env, alpha, gamma)
            print(f"Training with alpha={alpha}, gamma={gamma}")
            rewards, steps, successes = agent.train()
            results[(alpha, gamma)] = (rewards, steps, successes)

        # Visualize each metric on its own figure as a 2x2 grid
    hyperparams = list(results.keys())
    metrics_names = ['Reward per Episode', 'Steps per Episode', 'Success Rate Over Time']
    metrics_data = [0, 1, 2]

    for metric_idx, title in zip(metrics_data, metrics_names):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(title, fontsize=16)

        for i, (alpha, gamma) in enumerate(hyperparams):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            data = results[(alpha, gamma)][metric_idx]
            # For success rate, compute cumulative average
            if metric_idx == 2:
                data = np.cumsum(data) / np.arange(1, len(data) + 1)

            ax.plot(data)
            ax.set_title(f"α={alpha}, γ={gamma}")
            ax.set_xlabel("Episode")
            ylabel = "Total Reward" if metric_idx == 0 else ("Steps" if metric_idx == 1 else "Success Rate")
            ax.set_ylabel(ylabel)
            ax.grid(True)

        # Hide any unused subplot (in case hyperparams != 4)
        if len(hyperparams) < 4:
            for j in range(len(hyperparams), 4):
                fig.delaxes(axes.flat[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

