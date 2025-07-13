import os, sys, shutil, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_SingleAgent import SimpleSingleAgentEnv
from dqn_agent import DQNAgent

EXPERIMENTS = [
    dict(label='A-base',    buffer_size=100_000, batch_size=64, eps_decay_steps=250_000),
    dict(label='B-miniB',   buffer_size=100_000, batch_size=32, eps_decay_steps=250_000),
    dict(label='C-smallRB', buffer_size=50_000,  batch_size=64, eps_decay_steps=250_000),
    dict(label='D-fastE',   buffer_size=100_000, batch_size=64, eps_decay_steps=125_000),
]

EPISODES  = 10000
MAX_STEPS = 100
RESULTS_DIR = Path('results')


def encode_state(obs, env):
    x, y     = obs
    ix, iy   = env.item_pos
    vx, vy   = env.victim_pos
    wx, wy   = env.wall_pos
    fx, fy   = env.fire_pos
    has_item = int(env.agent_has_item)
    return np.array([x, y, ix, iy, vx, vy, wx, wy, fx, fy, has_item], dtype=np.float32)


def train_one_run(label: str, buffer_size: int, batch_size: int, eps_decay_steps: int):
    env = SimpleSingleAgentEnv(size=5, randomize=False)
    state_dim, n_actions = 11, 4
    agent = DQNAgent(
        state_dim,
        n_actions,
        buffer_size=buffer_size,
        batch_size=batch_size,
        eps_decay_steps=eps_decay_steps,
        target_update_freq=1000,
        device=torch.device('cpu')
    )

    metrics = {k: [] for k in ['Reward', 'Length', 'Success', 'Collisions', 'Fires', 'Loss', 'Epsilon']}

    for ep in range(1, EPISODES+1):
        obs = env.reset()
        state = encode_state(obs, env)
        ep_R = collisions = fires = 0

        for t in range(1, MAX_STEPS+1):
            action = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = encode_state(next_obs, env)

            loss = agent.step((state, action, reward, next_state, done))
            state = next_state
            ep_R += reward
            collisions += int(reward == -1)
            fires += int(reward == -10)

            if loss is not None:
                metrics['Loss'].append(loss)
                metrics['Epsilon'].append(agent.eps)

            if done:
                break

        metrics['Reward'].append(ep_R)
        metrics['Length'].append(t)
        metrics['Success'].append(int(ep_R > 0 and fires == 0))
        metrics['Collisions'].append(collisions)
        metrics['Fires'].append(fires)

        if ep % 1000 == 0 or ep == 1:
            print(f"{label}: Ep {ep}/{EPISODES} | R={ep_R:.1f} | ε={agent.eps:.3f}")

    # Save results
    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ep = pd.DataFrame({
        'Episode': range(1, EPISODES+1),
        'Reward': metrics['Reward'],
        'Length': metrics['Length'],
        'Success': metrics['Success'],
        'Collisions': metrics['Collisions'],
        'Fires': metrics['Fires']
    })
    df_ep.to_csv(out_dir / 'episode_metrics.csv', index=False)

    df_le = pd.DataFrame({
        'Loss': metrics['Loss'],
        'Epsilon': metrics['Epsilon']
    })
    df_le.to_csv(out_dir / 'loss_eps.csv', index=False)

    torch.save(agent.online_net.state_dict(), out_dir / 'weights.pth')

    return df_ep.assign(Label=label)


def main():
    # Clean results if requested
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
        sys.exit(0)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_runs = []
    for cfg in EXPERIMENTS:
        print(f"\n=== RUN {cfg['label']} ===")
        all_runs.append(train_one_run(**cfg))

    summary = pd.concat(all_runs, ignore_index=True)
    summary.to_csv(RESULTS_DIR / 'summary_all.csv', index=False)


if __name__ == '__main__':
    main()