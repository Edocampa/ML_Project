import numpy as np
import pandas as pd
import torch
from pathlib import Path


from env_MultiAgent import SimpleGridWorld     
from dqn_agent import DQNAgent                      

# Definition of hyperparameters

BUFFER_SIZE = 10_000
BATCH_SIZE  = 32
EPS_DECAY   = 100_000
EPISODES    = 25000
MAX_STEPS   = 100
RESULTS_DIR = Path('results')

# common dict that include all hyparameters of the agents

AGENT_CFG = dict(buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 eps_decay_steps=EPS_DECAY,
                 device=torch.device('cpu'))

from env_MultiAgent import SimpleGridWorld  # ambiente 2‑agenti
from dqn_agent import DQNAgent              


STATE_DIM = 13   # 11 feature base + other agent coordinates
ACTIONS_N = 4   


# Encoding of the states

def encode_state(self_obs, other_obs, env, agent_id):

    x,  y  = self_obs # agent 0
    ox, oy = other_obs # agent 1

    x,  y  = self_obs
    ox, oy = other_obs


    ix, iy = env.item_pos
    vx, vy = env.victim_pos
    wx, wy = env.wall_pos
    fx, fy = env.fire_pos

    has_item = getattr(env, f"agent{agent_id}_has_item", False)

    return np.array([
        x,  y,  ox, oy,             # agent coordinates
        ix, iy, vx, vy,             # ostatic items
        wx, wy, fx, fy,             # wall and fire
        int(has_item)               # flag if agent pick up item
    ], dtype=np.float32)


STATE_DIM  = 13 # 11 + 2 (coordinates other agent)
ACTIONS_N  = 4

# ───────── training loop ─────────
def main():

    # Definition of env and 2 instance of DQNAgent for the 2 agents

    env     = SimpleGridWorld(size=5, randomize=False)
    agents  = [DQNAgent(STATE_DIM, ACTIONS_N, **AGENT_CFG) for _ in range(2)]

    # Saved metrics

    metrics = {i: dict(Reward=[], Success=[]) for i in range(2)}

    # Training Loop

    for ep in range(EPISODES):
        obs_list = env.reset()   # [obs0, obs1], one per each agent
        states   = [encode_state(obs_list[i], obs_list[1-i], env, i)
                    for i in range(2)]

        ep_R = [0.0, 0.0]      # cumulative reward per episode
        succ = [0,   0]        # success episode flag

    # Step Loop

        for _ in range(MAX_STEPS):

            # Selection of the action and step in the env



# Training loop 
def train_one_run(
    episodes: int = 25_000,
    max_steps: int = 100,
    buffer_size: int = 10_000,
    batch_size: int = 32,
    eps_decay: int = 100_000,
    save_dir: Path | str = "results",
    device: torch.device | str = torch.device("cpu"),
    env_size: int = 5,
    randomize_env: bool = False,
):

    save_dir = Path(save_dir)
    env = SimpleGridWorld(size=env_size, randomize=randomize_env)

    common_cfg = dict(buffer_size=buffer_size,
                      batch_size=batch_size,
                      eps_decay_steps=eps_decay,
                      device=device)
    
    # Creation of 2 instances of DQNAgent

    agents = [DQNAgent(STATE_DIM, ACTIONS_N, **common_cfg) for _ in range(2)]

    metrics = {i: {"Reward": [], "Success": []} for i in range(2)}

    for ep in range(episodes):
        obs_list = env.reset()
        states = [encode_state(obs_list[i], obs_list[1 - i], env, i)
                  for i in range(2)]

        ep_R = [0.0, 0.0]
        succ = [0, 0]

        # Step  Loop

        for t in range(1, max_steps + 1):

             # Selection of the action and step in the env

            actions = [agents[i].select_action(states[i]) for i in range(2)]

            next_obs_list, rewards, done, info = env.step(actions)

            next_states = [encode_state(next_obs_list[i], next_obs_list[1 - i], env, i)
                           for i in range(2)]
            
            # Store transition and optimize each agent

           # Store transition and optimize each agent
           
            for i in range(2):

                agents[i].step((states[i], actions[i], rewards[i],
                                 next_states[i], done))


                agents[i].step((states[i], actions[i], rewards[i], next_states[i], done))

                ep_R[i] += rewards[i]
                states[i] = next_states[i]

            if done:
                break

        if "success" in info:
            succ = [int(x) for x in info["success"]]
        else:
            succ = [int(r > 0) for r in ep_R]

        for i in range(2):

            if 'success' in info:
                succ[i] = int(info['success'][i])
            else:
                succ[i] = int(ep_R[i] > 0)      
            metrics[i]['Reward'].append(ep_R[i])
            metrics[i]['Success'].append(succ[i])

            metrics[i]["Reward"].append(ep_R[i])
            metrics[i]["Success"].append(succ[i])


        if (ep + 1) % 1000 == 0 or ep == 0:
            print(
                f"Ep {ep + 1}/{episodes} | "
                f"R0={ep_R[0]:5.1f} R1={ep_R[1]:5.1f} | "
                f"ε0={agents[0].eps:.3f}"
            )


    save_dir.mkdir(exist_ok=True)
    for i in range(2):
        pd.DataFrame(metrics[i]).to_csv(save_dir / f"agent{i}_metrics.csv", index=False)
        torch.save(agents[i].online_net.state_dict(), save_dir / f"agent{i}_weights.pth")

    return metrics

def main():
    train = train_one_run()


if __name__ == "__main__":
    main()

