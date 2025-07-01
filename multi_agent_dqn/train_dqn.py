import numpy as np
import torch
from env import SimpleGridWorld
from dqn_agent import DQNAgent

def encode_state(obs, env):
    # [(x1,y1),(x2,y2)] + flag item
    (x1,y1),(x2,y2) = obs
    return np.array([x1,y1, x2,y2,
                     int(env.agent1_has_item),
                     int(env.agent2_has_item)], dtype=np.float32)

def main():
    # Iperparametri
    episodes = 500
    max_steps = 200

    # Crea env
    env = SimpleGridWorld(size=5, randomize=True)
    state_dim = 6
    n_actions = 4
    agent = DQNAgent(state_dim, n_actions, device=torch.device('cpu'))

    for ep in range(1, episodes+1):
        obs = env.reset()
        state = encode_state(obs, env)
        total_rewards = [0.0, 0.0]   # [agente1, agente2]

        # ------- INNER LOOP DENTRO ep LOOP -------
        for t in range(max_steps):
            a = agent.select_action(state)
            next_obs, (r1, r2), done, _ = env.step([a, a])
            next_state = encode_state(next_obs, env)

            # apprendimento su r1
            agent.learn((state, a, r1, next_state, done))

            state = next_state
            total_rewards[0] += r1
            total_rewards[1] += r2

            if done:
                break

        # ------- PRINT DENTRO ep LOOP -------
        print(
            f"Ep {ep:4d} | "
            f"Reward A1 {total_rewards[0]:6.1f} | "
            f"Reward A2 {total_rewards[1]:6.1f} | "
            f"ε {agent.eps:.2f}"
        )

    # salva pesi alla fine di tutti gli episodi
    torch.save(agent.online_net.state_dict(), "dqn_weights.pth")


if __name__ == "__main__":
    main()
