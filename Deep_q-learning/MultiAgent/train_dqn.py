import torch
from pathlib import Path
from env_MultiAgent import SimpleGridWorld 
from dqn_agent import DQNAgent
import numpy as np
import pandas as pd

STATE_DIM = 13   # 11 feature base + other agent coordinates
ACTIONS_N = 4

# Encoding of the states

def encode_state(self_obs, other_obs, env, agent_id):
    x,  y  = self_obs
    ox, oy = other_obs

    ix, iy = env.item_pos
    vx, vy = env.victim_pos
    wx, wy = env.wall_pos
    fx, fy = env.fire_pos

    has_item = getattr(env, f"agent{agent_id}_has_item", False)

    return np.array([
        x,  y,  ox, oy,            # agent & other agent
        ix, iy, vx, vy,            # item & victim
        wx, wy, fx, fy,            # wall & fire
        int(has_item)              # possession flag
    ], dtype=np.float32)

# Training loop 

def train_one_run(
    episodes: int = 25000,
    max_steps: int = 100,
    buffer_size: int = 10000,
    batch_size: int = 32,
    eps_decay: int = 100000,
    save_dir: Path | str = "results",
    device: torch.device | str = torch.device("cpu"),
):
    save_dir = Path(save_dir)
    env = SimpleGridWorld(size=5, randomize=False)

    # Dict used for shared parameters

    common_cfg = dict(
        buffer_size=buffer_size,
        batch_size=batch_size,
        eps_decay_steps=eps_decay,
        device=device
    )

    # Create 2 instances of DQNAgent
    # ** is equivalent to call 2 times DQNAgent with parameters included in common_cfg

    agents = [DQNAgent(STATE_DIM, ACTIONS_N, **common_cfg) for _ in range(2)]

    metrics = {i: {"Reward": [], "Success": []} for i in range(2)}

    for ep in range(episodes):
        obs_list = env.reset()
        states = [
            encode_state(obs_list[i], obs_list[1 - i], env, i)
            for i in range(2)
        ]

        ep_R = [0.0, 0.0]

        # Step  Loop

        for t in range(1, max_steps + 1):

            # Selection of the action and step in the env
            actions = [agents[i].select_action(states[i]) for i in range(2)]

        
            next_obs_list, rewards, done, _ = env.step(actions)
            

            next_states = [
                encode_state(next_obs_list[i], next_obs_list[1 - i], env, i)
                for i in range(2)
            ]

             # Store transition and optimize each agent

            for i in range(2):
                agents[i].step((states[i], actions[i], rewards[i], next_states[i], done))
                ep_R[i] += rewards[i]
                states[i] = next_states[i]

            if done:
                break

        succ = [int(r > 0) for r in ep_R]

        for i in range(2):
            metrics[i]["Reward"].append(ep_R[i])
            metrics[i]["Success"].append(succ[i])

        if (ep + 1) % 1000 == 0 or ep == 0:
            print(
                f"Ep {ep + 1}/{episodes} | "
                f"R0={ep_R[0]:5.1f} R1={ep_R[1]:5.1f} | "
                f"S0={succ[0]} S1={succ[1]} | "
                f"ε0={agents[0].eps:.3f}"
            )

    save_dir.mkdir(exist_ok=True)
    for i in range(2):
        pd.DataFrame(metrics[i]).to_csv(save_dir / f"agent{i}_metrics.csv", index=False)
        torch.save(agents[i].online_net.state_dict(), save_dir / f"agent{i}_weights.pth")

    return metrics

def main():
    train_one_run()

if __name__ == "__main__":
    main()
