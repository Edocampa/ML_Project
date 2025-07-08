import os
import random
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from env_SingleAgent import SimpleSingleAgentEnv
from dqn_agent import DQNAgent

# Experiment configurations
EXPERIMENTS = [
    dict(label='A-base',    buffer_size=100_000, batch_size=64,  eps_decay_steps=800_000),
    dict(label='B-smallB',  buffer_size=100_000, batch_size=32,  eps_decay_steps=800_000),
    dict(label='C-smallRB', buffer_size=10_000,   batch_size=64,  eps_decay_steps=200_000),
    dict(label='D-fastE',   buffer_size=100_000, batch_size=64,  eps_decay_steps=200_000),
]
EPISODES = 10000
MAX_STEPS = 200
RESULTS_DIR = Path('results_10x10')

# State encoding
def encode_state(obs, env):
    ax, ay = obs
    ix, iy = env.item_pos
    vx, vy = env.victim_pos
    wx, wy = env.wall_pos
    fx, fy = env.fire_pos
    has_item = int(env.agent_has_item)
    return np.array([ax, ay, ix, iy, vx, vy, wx, wy, fx, fy, has_item], dtype=np.float32)


def train_one_run(cfg):
    label = cfg['label']
    print(f"\n=== RUN {label} ===")
    env = SimpleSingleAgentEnv(size=10, randomize=True)
    state_dim = 11
    n_actions = 4
    agent = DQNAgent(
        state_dim, n_actions,
        buffer_size=cfg['buffer_size'],
        batch_size=cfg['batch_size'],
        eps_decay_steps=cfg['eps_decay_steps'],
        target_update_freq=2000,
        device=torch.device('cpu')
    )

    metrics = {k: [] for k in ['Reward','Length','Success','Collisions','Fires','Loss','Epsilon']}

    for ep in range(1, EPISODES+1):
        obs = env.reset()
        state = encode_state(obs, env)
        ep_reward, collisions, fires = 0, 0, 0

        for t in range(1, MAX_STEPS+1):
            action = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = encode_state(next_obs, env)
            loss = agent.step((state, action, reward, next_state, done))

            state = next_state
            ep_reward += reward
            collisions += int(reward == -1)
            fires += int(reward == -10)
            if loss is not None:
                metrics['Loss'].append(loss)
                metrics['Epsilon'].append(agent.eps)
            if done:
                break

        metrics['Reward'].append(ep_reward)
        metrics['Length'].append(t)
        metrics['Success'].append(int(ep_reward > 0 and fires == 0))
        metrics['Collisions'].append(collisions)
        metrics['Fires'].append(fires)

        if ep % 1000 == 0 or ep == 1:
            print(f"{label}: Ep {ep}/{EPISODES} | R={ep_reward:.1f} | eps={agent.eps:.3f}")

    # save metrics
    out_dir = RESULTS_DIR / label
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ep = pd.DataFrame({
        'Episode': range(1, EPISODES+1),
        'Reward': metrics['Reward'],
        'Length': metrics['Length'],
        'Success': metrics['Success'],
        'Collisions': metrics['Collisions'],
        'Fires': metrics['Fires'],
    })
    df_ep.to_csv(out_dir/'episode_metrics.csv', index=False)

    df_le = pd.DataFrame({'Loss': metrics['Loss'], 'Epsilon': metrics['Epsilon']})
    df_le.to_csv(out_dir/'loss_eps.csv', index=False)

    agent.save(out_dir/'weights.pth')
    return df_ep.assign(Label=label)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
        sys.exit(0)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_runs = []
    for cfg in EXPERIMENTS:
        df_run = train_one_run(cfg)
        all_runs.append(df_run)
    summary = pd.concat(all_runs, ignore_index=True)
    summary.to_csv(RESULTS_DIR/'summary_all.csv', index=False)

if __name__ == '__main__':
    main()