import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.ND_env_SingleAgent_10x10 import SimpleSingleAgentEnv

# Q-Learning hyperparameters
# Note alpha now computed dynamically via visit counts
GAMMAS = [0.15, 0.99]
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 6000
MAX_STEPS_PER_EPISODE = 200
N_ACTIONS = 4

SMOOTH_WINDOW = 50
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW


def state_to_index(env):
    """
    Maps environment state to a hashable index for Q-table keys.
    Important to represent the state into the Q-table.
    """
    x, y = env.agent_pos
    has_item = int(env.agent_has_item)
    return (x, y, has_item)


class TabularQLearnerStochastic:
    """
    Tabular Q-Learning for non-deterministic (stochastic) environments,
    using dynamic step-size α = 1 / (1 + visits(s,a)).
    """
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.n_actions = N_ACTIONS
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        # Visit counts for (s,a)
        self.visits = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.int32))

    def choose_action(self, state, epsilon):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def train(self):
        rewards_per_episode = [] #total reward per episode
        steps_per_episode = [] # number of step for complete the episode
        success_per_episode = [] #1 if success episode 0 otherwise
        epsilon = EPSILON_START

        for ep in range(1, NUM_EPISODES + 1):
            self.env.reset()
            state = state_to_index(self.env)
            total_reward = 0.0
            steps = 0
            success = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                #choose action a
                action = self.choose_action(state, epsilon)
                #execute action a
                _, reward, done, _ = self.env.step(action)
                #hash new state
                next_state = state_to_index(self.env)

                # increment visit count and compute dynamic alpha
                self.visits[state][action] += 1
                alpha = 1.0 / (1 + self.visits[state][action])

                # Q-learning update with dynamic α
                old = self.Q[state][action]
                target = reward + (0.0 if done else self.gamma * np.max(self.Q[next_state]))
                self.Q[state][action] = (1 - alpha) * old + alpha * target

                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    if reward >= 10:
                        success = True
                    break

            # decay epsilon
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
            success_per_episode.append(1 if success else 0)

        return rewards_per_episode, steps_per_episode, success_per_episode

if __name__ == '__main__':
    results = {}

    for gamma in GAMMAS:
        print(f"Training stochastic γ={gamma} …")
        env = SimpleSingleAgentEnv(size=10, randomize=False)
        learner = TabularQLearnerStochastic(env, gamma)
        
        start = time.time()
        rewards, steps, successes = learner.train()
        elapsed = time.time() - start
        print(f" → Training took {elapsed:.2f}s")
                
        cum_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        results[gamma] = {
            'reward': np.array(rewards),
            'steps': np.array(steps),
            'success': cum_success
        }

    # Plotting metrics
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
