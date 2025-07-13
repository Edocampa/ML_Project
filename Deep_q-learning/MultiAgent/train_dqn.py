import numpy as np
import pandas as pd
import torch
from pathlib import Path

from env_MultiAgent import SimpleGridWorld     
from dqn_agent import DQNAgent                      

# ───────── config ─────────
BUFFER_SIZE = 100_000
BATCH_SIZE  = 64
EPS_DECAY   = 250_000
EPISODES    = 25000
MAX_STEPS   = 100
RESULTS_DIR = Path('results')

AGENT_CFG = dict(buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 eps_decay_steps=EPS_DECAY,
                 device=torch.device('cpu'))

# ───────── state encoder ─────────
def encode_state(self_obs, other_obs, env, agent_id):
    # assumo self_obs/other_obs = (x, y)
    x,  y  = self_obs
    ox, oy = other_obs

    ix, iy = env.item_pos
    vx, vy = env.victim_pos
    wx, wy = env.wall_pos
    fx, fy = env.fire_pos

    has_item = getattr(env, f'agent{agent_id}_has_item', False)

    return np.array([x, y, ox, oy,
                     ix, iy, vx, vy,
                     wx, wy, fx, fy,
                     int(has_item)], dtype=np.float32)

STATE_DIM  = 13
ACTIONS_N  = 4

# ───────── training loop ─────────
def main():
    env     = SimpleGridWorld(size=5, randomize=False)
    agents  = [DQNAgent(STATE_DIM, ACTIONS_N, **AGENT_CFG) for _ in range(2)]
    metrics = {i: dict(Reward=[], Success=[]) for i in range(2)}

    for ep in range(EPISODES):
        obs_list = env.reset()                              # [obs0, obs1]
        states   = [encode_state(obs_list[i], obs_list[1-i], env, i)
                    for i in range(2)]

        ep_R = [0.0, 0.0]      # reward cumulativa episodio
        succ = [0,   0]        # flag successo episodio

        for _ in range(MAX_STEPS):
            actions = [agents[i].select_action(states[i]) for i in range(2)]
            next_obs_list, rewards, done, info = env.step(actions)

            next_states = [encode_state(next_obs_list[i], next_obs_list[1-i],
                                         env, i)
                           for i in range(2)]

            for i in range(2):
                agents[i].step((states[i], actions[i], rewards[i],
                                 next_states[i], done))

                ep_R[i] += rewards[i]
                states[i] = next_states[i]

            if done:
                break

        # success flags
        for i in range(2):
            if 'success' in info:
                succ[i] = int(info['success'][i])
            else:
                succ[i] = int(ep_R[i] > 0)      # fallback
            metrics[i]['Reward'].append(ep_R[i])
            metrics[i]['Success'].append(succ[i])

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(f"Ep {ep+1}/{EPISODES} | "
                  f"R0={ep_R[0]:5.1f} R1={ep_R[1]:5.1f} | "
                  f"ε0={agents[0].eps:.3f}")

    # ─────── save ───────
    RESULTS_DIR.mkdir(exist_ok=True)
    for i in range(2):
        pd.DataFrame(metrics[i]).to_csv(
            RESULTS_DIR / f'agent{i}_metrics.csv', index=False)
        torch.save(agents[i].online_net.state_dict(),
                   RESULTS_DIR / f'agent{i}_weights.pth')

    print("✓ Training multi-agent completato → results_multi/")

if __name__ == "__main__":
    main()
