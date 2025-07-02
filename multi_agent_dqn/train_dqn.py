import numpy as np
from env_SingleAgent import SimpleSingleAgentEnv
from dqn_agent import DQNAgent

def encode_state(obs, env):
    # obs is (x, y)
    x, y = obs
    has_item = int(env.agent_has_item)
    # input: [x, y, has_item]
    return np.array([x, y, has_item], dtype=np.float32)

def main():
    env = SimpleSingleAgentEnv(size=5, randomize=True)
    state_dim = 3
    n_actions = 4
    agent = DQNAgent(state_dim, n_actions)

    n_episodes = 500
    max_steps = 200

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        state = encode_state(obs, env)
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = encode_state(next_obs, env)

            agent.learn((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {ep:4d} | Total Reward: {total_reward:6.1f} | Epsilon: {agent.eps:.3f}")

    # Save model
    import torch
    torch.save(agent.net.state_dict(), "dqn_single_agent.pth")

if __name__ == "__main__":
    main()