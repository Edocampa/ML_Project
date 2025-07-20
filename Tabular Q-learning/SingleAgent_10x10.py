import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_singleAgent_10dim import SimpleSingleAgentEnv

# Q-Learning hyperparameters (deterministic pure backup rule)
GAMMAS = [0.15, 0.99]
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200  # increased for larger grid
N_ACTIONS = 4

SMOOTH_WINDOW = 50
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW


def state_to_index(env):
    """
    Hashable representation of state: (x, y, has_item).
    Important to represent the state into the Q-table.
    """
    x, y = env.agent_pos
    has_item = int(env.agent_has_item)
    return (x, y, has_item)


class TabularQLearnerDeterministic:
    """
    Tabular Q-Learning for deterministic environments,
    using pure backup rule: Q[s,a] ← r + (gamma) max_a' Q[s',a'].
    """
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.n_actions = N_ACTIONS
        # Q-table: init with 0 (best practice)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
    #epsilon-greedy policy 
    def choose_action(self, state, epsilon):
        #exploration
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        #exploitation
        return int(np.argmax(self.Q[state]))

    def train(self):
        rewards = [] #total reward per episode
        steps = []  # number of step for complete the episode
        successes = []  #1 if success episode 0 otherwise
        epsilon = EPSILON_START

        for ep in range(1, NUM_EPISODES + 1):
            self.env.reset()
            state = state_to_index(self.env)
            total_reward = 0.0
            step_count = 0
            success = False

            for _ in range(MAX_STEPS_PER_EPISODE):
                #choose action a
                action = self.choose_action(state, epsilon)
                #execute action a
                _, reward, done, _ = self.env.step(action)
                #hash new state
                next_state = state_to_index(self.env)

                # pure backup update
                self.Q[state][action] = reward + (0.0 if done else self.gamma * np.max(self.Q[next_state]))

                state = next_state
                total_reward += reward
                step_count += 1

                if done:
                    success = (reward >= 10)
                    break

            rewards.append(total_reward)
            steps.append(step_count)
            successes.append(int(success))
            #decrementa epsilon 
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        return np.array(rewards), np.array(steps), np.array(successes)

if __name__ == '__main__':
    results = {}

    for gamma in GAMMAS:
        print(f"Training deterministic 10×10 γ={gamma} …")
        env = SimpleSingleAgentEnv(size=10, randomize=False)
        learner = TabularQLearnerDeterministic(env, gamma)

        start_time = time.time()
        rewards, steps, successes = learner.train()
        elapsed = time.time() - start_time
        print(f" → Training took {elapsed:.2f}s")

        cum_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
        results[gamma] = {
            'reward': rewards,
            'steps': steps,
            'success': cum_success
        }

    metrics = [
        ('reward',  'Total Reward per Episode'),
        ('steps',   'Steps per Episode'),
        ('success', 'Cumulative Success Rate')
    ]

    for key, title in metrics:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        fig.suptitle(f"Deterministic Q-Learning 10x10: {title}", fontsize=16)

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

