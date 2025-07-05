import os
import numpy as np
import torch
import pandas as pd
from env_SingleAgent import SimpleSingleAgentEnv, ITEM
from dqn_agent import DQNAgent

# ── Iper-parametri da variare ─────────────────────────────
BUFFER_SIZE = 100000   # dimensione del replay buffer
BATCH_SIZE  = 64       # dimensione del minibatch
EPS_DECAY   = 1e-6     # decay di epsilon (esplorazione)
EPISODES    = 500      # numero di episodi di training
MAX_STEPS   = 200      # passi massimi per episodio
# ──────────────────────────────────────────────────────────

# Storage for metrics
rewards    = []
lengths    = []
successes  = []
collisions = []
fires      = []
losses     = []
epsilons   = []

def encode_state(obs, env):
    x, y     = obs
    ix, iy   = env.item_pos
    vx, vy   = env.victim_pos
    wx, wy   = env.wall_pos
    fx, fy   = env.fire_pos
    has_item = int(env.agent_has_item)
    return np.array([
        x,  y,
        ix, iy,
        vx, vy,
        wx, wy,
        fx, fy,
        has_item
    ], dtype=np.float32)

# Main training loop
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    env       = SimpleSingleAgentEnv(size=5, randomize=True)
    state_dim = 11  
    n_actions = 4
    agent     = DQNAgent(
        state_dim,
        n_actions,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        eps_decay=EPS_DECAY,
        device=torch.device('cpu')
    )

    for ep in range(1, EPISODES+1):
        obs   = env.reset()
        state = encode_state(obs, env)
        R = col = fire = 0

        for t in range(MAX_STEPS):
            a = agent.select_action(state)
            next_obs, r, done, _ = env.step(a)
            next_state = encode_state(next_obs, env)

            loss = agent.learn((state, a, r, next_state, done))
            if loss is not None:
                losses.append(loss)
                epsilons.append(agent.eps)

            state = next_state
            R    += r
            if r == -1:   col  += 1
            if r == -10:  fire += 1
            if done:
                break

        # Append episode metrics
        rewards.append(R)
        lengths.append(t+1)
        successes.append(int(R > 0 and fire == 0))
        collisions.append(col)
        fires.append(fire)

        print(f"Ep {ep:4d} | Reward {R:6.1f} | eps {agent.eps:.3f}")

    # Save model
    torch.save(agent.online_net.state_dict(), "results/dqn_weights.pth")

    # Save episode metrics (rinomina CSV dopo, vedi istruzioni)
    df = pd.DataFrame({
        'Reward':     rewards,
        'Length':     lengths,
        'Success':    successes,
        'Collisions': collisions,
        'Fires':      fires
    })
    df.to_csv("results/dqn_results.csv", index=False)

    # Save loss & epsilon over batches
    pd.DataFrame({'Loss': losses, 'Epsilon': epsilons}) \
        .to_csv("results/loss_eps.csv", index=False)