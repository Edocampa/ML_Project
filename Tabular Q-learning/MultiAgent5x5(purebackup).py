import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys, os

# Add project root to path so we can import our environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.env_MultiAgent import SimpleGridWorld

# Hyperparameters (pure backup rule, no alpha)
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200
N_ACTIONS = 4

SMOOTH_WINDOW = 50
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW


def state_to_index(env: SimpleGridWorld):
    # Encode both agents' positions and possession flags
    x1, y1 = env.agent1_pos
    x2, y2 = env.agent2_pos
    h1 = int(env.agent1_has_item)
    h2 = int(env.agent2_has_item)
    return (x1, y1, x2, y2, h1, h2)

if __name__ == '__main__':
    # Independent Q-learning with pure backup updates
    Q1 = defaultdict(lambda: np.zeros(N_ACTIONS))
    Q2 = defaultdict(lambda: np.zeros(N_ACTIONS))
    eps1 = EPSILON_START
    eps2 = EPSILON_START

    rewards1, rewards2, steps_log, success_log = [], [], [], []

    start_all = time.time()
    for ep in range(1, NUM_EPISODES + 1):
        env = SimpleGridWorld(size=5, randomize=False)
        env.reset()
        state = state_to_index(env)
        total_r1 = total_r2 = 0.0
        steps = 0
        success = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            # ε-greedy actions
            a1 = np.random.randint(N_ACTIONS) if np.random.rand() < eps1 else int(np.argmax(Q1[state]))
            a2 = np.random.randint(N_ACTIONS) if np.random.rand() < eps2 else int(np.argmax(Q2[state]))

            _, (r1, r2), done, _ = env.step([a1, a2])
            next_state = state_to_index(env)

            # pure backup updates: Q = r + γ max Q(next)
            Q1[state][a1] = r1 + (0.0 if done else GAMMA * np.max(Q1[next_state]))
            Q2[state][a2] = r2 + (0.0 if done else GAMMA * np.max(Q2[next_state]))

            state = next_state
            total_r1 += r1
            total_r2 += r2
            steps += 1

            if done:
                success = (r1 > 0 and r2 > 0)
                break

        eps1 = max(MIN_EPSILON, eps1 * EPSILON_DECAY)
        eps2 = max(MIN_EPSILON, eps2 * EPSILON_DECAY)

        rewards1.append(total_r1)
        rewards2.append(total_r2)
        steps_log.append(steps)
        success_log.append(int(success))

        if ep % 1000 == 0:
            print(f"Episode {ep:5d} | R1={np.mean(rewards1[-1000:]):.2f} R2={np.mean(rewards2[-1000:]):.2f} "
                  f"Success={np.mean(success_log[-1000:]):.2f}")

    print(f"Training completed in {time.time() - start_all:.2f} sec")

    # Plot metrics
    success_rate = np.cumsum(success_log) / np.arange(1, NUM_EPISODES + 1)
    data = {
        'Agent1 Reward': np.array(rewards1),
        'Agent2 Reward': np.array(rewards2),
        'Steps': np.array(steps_log),
        'Success Rate': success_rate
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pure Backup Independent Q-Learning (Multi-Agent)', fontsize=18)

    for ax, (label, arr) in zip(axes.flat, data.items()):
        ax.plot(arr, color='lightgray', label='raw')
        if len(arr) >= SMOOTH_WINDOW:
            smooth = np.convolve(arr, kernel, mode='valid')
            x = np.arange(SMOOTH_WINDOW - 1, len(arr))
            ax.plot(x, smooth, label=f'{SMOOTH_WINDOW}-ep MA', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Episode')
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
