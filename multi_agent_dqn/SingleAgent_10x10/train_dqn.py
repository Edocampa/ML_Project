import os, sys, shutil, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_SingleAgent import SimpleSingleAgentEnv
from dqn_agent import DQNAgent

# ───────────────────────── hyper‑grid ──────────────────────────
EXPERIMENTS = [
    dict(label='A-base',    buffer_size=100_000, batch_size=64, eps_decay=1e-6),
    dict(label='B-miniB',   buffer_size=100_000, batch_size=32, eps_decay=1e-6),
    dict(label='C-smallRB', buffer_size=10_000,  batch_size=64, eps_decay=1e-6),
    dict(label='D-fastE',   buffer_size=100_000, batch_size=64, eps_decay=5e-6),
]

EPISODES   = 5000    
MAX_STEPS  = 400    
RESULTS_DIR = Path('results')

# ───────────────────────── state encoder ──────────────────────────

def encode_state(obs, env):
    x, y     = obs
    ix, iy   = env.item_pos
    vx, vy   = env.victim_pos
    wx, wy   = env.wall_pos
    fx, fy   = env.fire_pos
    has_item = int(env.agent_has_item)
    return np.array([x, y, ix, iy, vx, vy, wx, wy, fx, fy, has_item], dtype=np.float32)

# ───────────────────────── training loop ──────────────────────────

def train_one_run(label, buffer_size, batch_size, eps_decay):
    env = SimpleSingleAgentEnv(size=10, randomize=True)
    state_dim, n_actions = 11, 4
    agent = DQNAgent(state_dim, n_actions, buffer_size=buffer_size, batch_size=batch_size,
                     eps_decay=eps_decay, device=torch.device('cpu'))

    metrics = dict(Reward=[], Length=[], Success=[], Collisions=[], Fires=[], Loss=[], Epsilon=[])

    for ep in range(EPISODES):
        obs = env.reset(); state = encode_state(obs, env)
        ep_R = col = fire = 0
        for t in range(MAX_STEPS):
            a = agent.select_action(state)
            next_obs, r, done, _ = env.step(a)
            next_state = encode_state(next_obs, env)
            loss = agent.learn((state, a, r, next_state, done))
            state = next_state; ep_R += r
            if r == -1: col += 1
            if r == -10: fire += 1
            if loss is not None:
                metrics['Loss'].append(loss); metrics['Epsilon'].append(agent.eps)
            if done: break
        metrics['Reward'].append(ep_R)
        metrics['Length'].append(t+1)
        metrics['Success'].append(int(ep_R > 0 and fire == 0))
        metrics['Collisions'].append(col); metrics['Fires'].append(fire)

        if (ep+1) % 1000 == 0 or ep == 0:
            print(f"{label}: Ep {ep+1:4d}/{EPISODES} | R={ep_R:6.1f} | ε={agent.eps:.3f}")

    out = RESULTS_DIR/label; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k:v for k,v in metrics.items() if k not in ['Loss','Epsilon']}).to_csv(out/'episode_metrics.csv', index=False)
    pd.DataFrame({'Loss':metrics['Loss'], 'Epsilon':metrics['Epsilon']}).to_csv(out/'loss_eps.csv', index=False)
    torch.save(agent.online_net.state_dict(), out/'weights.pth')

    return pd.DataFrame({'Episode':range(1,EPISODES+1), 'Reward':metrics['Reward'],
                         'Success':metrics['Success'], 'Label':label})

# ───────────────────────── main ──────────────────────────

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
        sys.exit(0)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_runs = []
    for cfg in EXPERIMENTS:
        print(f"\n=== RUN {cfg['label']} ===")
        all_runs.append(train_one_run(**cfg))

    summary = pd.concat(all_runs, ignore_index=True)
    summary.to_csv(RESULTS_DIR/'summary_all.csv', index=False)

if __name__ == '__main__':
    main()
