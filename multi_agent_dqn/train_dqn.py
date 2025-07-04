import os
import numpy as np
import torch
import pandas as pd
from env_SingleAgent import SimpleSingleAgentEnv,ITEM
from dqn_agent import DQNAgent

# Storage for metrics
rewards   = []
lengths   = []
successes = []
collisions= []
fires     = []
losses    = []
epsilons  = []

# State encoding
def encode_state(obs, env):
    size = env.size
    # inizializzo zero tensor shape (5, size, size)
    grid = np.zeros((5, size, size), dtype=np.float32)

    # canale agent
    x,y = obs
    grid[0, x, y] = 1.0

    # canale item (se presente ancora)
    ix,iy = env.item_pos
    if env.grid[ix, iy] == ITEM:
        grid[1, ix, iy] = 1.0

    # canale victim
    vx,vy = env.victim_pos
    grid[2, vx, vy] = 1.0

    # canale wall
    wx,wy = env.wall_pos
    grid[3, wx, wy] = 1.0

    # canale fire
    fx,fy = env.fire_pos
    grid[4, fx, fy] = 1.0

    # flatten e ritorna
    return grid.flatten()

# Main training loop
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    env       = SimpleSingleAgentEnv(size=5, randomize=True)
    state_dim = 5 * env.size * env.size
    n_actions = 4
    agent     = DQNAgent(state_dim, n_actions, device=torch.device('cpu'))

    episodes  = 500
    max_steps = 200

    for ep in range(1, episodes+1):
        obs   = env.reset()
        state = encode_state(obs, env)
        R = col = fire = 0

        for t in range(max_steps):
            a = agent.select_action(state)
            next_obs, r, done, _ = env.step(a)
            next_state = encode_state(next_obs, env)

            loss = agent.learn((state, a, r, next_state, done))
            if loss is not None:
                losses.append(loss)
                epsilons.append(agent.eps)

            state = next_state
            R    += r
            if r == -1:  col += 1
            if r == -10: fire+= 1
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

    # Save episode metrics
    df = pd.DataFrame({
        'Reward':    rewards,
        'Length':    lengths,
        'Success':   successes,
        'Collisions':collisions,
        'Fires':     fires
    })
    df.to_csv("results/dqn_results.csv", index=False)

    # Save loss & epsilon over batches
    pd.DataFrame({'Loss': losses, 'Epsilon': epsilons}).to_csv("results/loss_eps.csv", index=False)
