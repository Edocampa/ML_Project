import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import time

# Add project root to path so we can import our environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_SingleAgent import SimpleSingleAgentEnv

# Q-Learning hyperparameters (pure backup rule)
GAMMAS = [0.15, 0.99]
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100
N_ACTIONS = 4

SMOOTH_WINDOW = 50
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW


def state_to_index(env):
    """
    Maps environment state to a hashable index for Q-table keys.
    """
    x, y = env.agent_pos
    has_item = int(env.agent_has_item)
    return (x, y, has_item)


class TabularQLearnerDeterministic:
    """
    Tabular Q-Learning for deterministic environments,
    using pure backup rule: Q[s,a] ← r + γ max_a' Q[s',a'].
    """
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.n_actions = N_ACTIONS
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    def choose_action(self, state, epsilon):
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def train(self):
        rewards = []
        steps = []
        successes = []
        epsilon = EPSILON_START

        for ep in range(1, NUM_EPISODES + 1):
            self.env.reset()
            state = state_to_index(self.env)
            total_reward = 0.0
            step_count = 0
            success = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = self.choose_action(state, epsilon)
                _, reward, done, _ = self.env.step(action)
                next_state = state_to_index(self.env)

                # pure backup update
                target = reward + (0.0 if done else self.gamma * np.max(self.Q[next_state]))
                self.Q[state][action] = target

                state = next_state
                total_reward += reward
                step_count += 1

                if done:
                    if reward >= 10:
                        success = True
                    break

            # record
            rewards.append(total_reward)
            steps.append(step_count)
            successes.append(success)

            # decay epsilon
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        return np.array(rewards), np.array(steps), np.array(successes, dtype=int)

    def evaluate(self, episodes=100):
        total_returns = []
        for _ in range(episodes):
            self.env.reset()
            state = state_to_index(self.env)
            ep_reward = 0.0
            done = False
            while not done:
                action = int(np.argmax(self.Q[state]))
                _, reward, done, _ = self.env.step(action)
                ep_reward += reward
                state = state_to_index(self.env)
            total_returns.append(ep_reward)
        return np.mean(total_returns)


if __name__ == '__main__':
    results = {}

    # Train for each gamma on the deterministic env
    for gamma in GAMMAS:
        print(f"Training deterministic γ={gamma} …")
        env = SimpleSingleAgentEnv(size=5, randomize=False)
        learner = TabularQLearnerDeterministic(env, gamma)

        start_time = time.time()
        rewards, steps, successes = learner.train()
        elapsed = time.time() - start_time
        print(f" → Training took {elapsed:.2f}s")

        # Cumulative success rate
        cum_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        results[gamma] = {
            'reward': rewards,
            'steps': steps,
            'success': cum_success
        }
    
        # Plot metrics in single-row layout
    metrics = [
        ('reward',  'Total Reward per Episode'),
        ('steps',   'Steps per Episode'),
        ('success', 'Cumulative Success Rate')
    ]

    for key, title in metrics:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        fig.suptitle(f"Stochastic Q-Learning: {title}", fontsize=16)

        for idx, (gamma, data) in enumerate(results.items()):
            ax = axes[idx] if len(results) > 1 else axes
            series = data[key]
            ax.plot(series, color='lightgray', label='raw')
            if len(series) >= SMOOTH_WINDOW:
                smooth = np.convolve(series, kernel, mode='valid')
                x = np.arange(SMOOTH_WINDOW - 1, len(series))
                ax.plot(x, smooth, label=f'{SMOOTH_WINDOW}-ep MA', linewidth=2)
            ax.set_title(f"γ={gamma}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(title)
            ax.grid(True)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
