from env.rescue_env_wrapper import RescueEnvWrapper

env = RescueEnvWrapper(render_mode="human")

env.reset()
for _ in range(100):
    for agent in env.agents:
        action = env.env.action_space(agent).sample()
        obs, reward, done, trunc, info = env.step(action)
        print(f"{agent} - Obs: {obs.shape}, Reward: {reward}")
env.close()