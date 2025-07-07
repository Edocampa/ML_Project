import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import time 
# adjust path so we can import your env class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_singleAgent_10dim import SimpleSingleAgentEnv

# Q-Learning hyperparameters
ALPHAS = [0.1, 0.5]
GAMMAS = [0.9, 0.99]
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200
N_ACTIONS = 4

# smoothing window (number of episodes)
SMOOTH_WINDOW = 50
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW


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
            success = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = self.choose_action(state, epsilon)
                _, reward, done, _ = self.env.step(action)
                next_state = state_to_index(self.env)

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
    # 1) Train for each (alpha, gamma)
    results = {}
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            print(f"Training α={alpha}, γ={gamma} on 10×10 grid…")
            env = SimpleSingleAgentEnv(size=10, randomize=False)
            agent = TabularQLearner(env, alpha, gamma)
            
            # ======== Profiling starts here ========
            start_time = time.time()
            rewards, steps, successes = agent.train()
            elapsed = time.time() - start_time
            print(f" → Training took {elapsed:.2f} seconds")
            # ======== Profiling ends here ========
            
            results[(alpha, gamma)] = {
                'reward': np.array(rewards),
                'steps': np.array(steps),
                'success': np.cumsum(successes) / np.arange(1, len(successes) + 1)
            }

    # 2) Plot each metric with smoothing
    metrics = [
        ('reward', 'Total Reward per Episode'),
        ('steps', 'Steps per Episode'),
        ('success', 'Cumulative Success Rate')
    ]

    for key, title in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"10×10 Grid: {title}", fontsize=18)

        for idx, ((alpha, gamma), data_dict) in enumerate(results.items()):
            row, col = divmod(idx, 2)
            ax = axes[row, col]
            data = data_dict[key]

            # raw trace (light gray)
            ax.plot(data, color='lightgray', label='raw')

            # smoothed trace
            if len(data) >= SMOOTH_WINDOW:
                smooth = np.convolve(data, kernel, mode='valid')
                x = np.arange(SMOOTH_WINDOW - 1, len(data))
                ax.plot(x, smooth, label=f'{SMOOTH_WINDOW}-ep MA', linewidth=2)

            ax.set_title(f"α={alpha}, γ={gamma}")
            ax.set_xlabel('Episode')
            ax.set_ylabel(title)
            ax.grid(True)
            ax.legend()

        # remove any unused subplot
        total_plots = len(ALPHAS) * len(GAMMAS)
        for j in range(total_plots, 4):
            fig.delaxes(axes.flat[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
