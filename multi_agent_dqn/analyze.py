import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results/dqn_results.csv')

# Compute rolling averages
window = 50
df['Reward_MA'] = df['Reward'].rolling(window).mean()
df['Success_MA'] = df['Success'].rolling(window).mean()

# Plot learning curve
plt.figure()
plt.plot(df['Reward_MA'], label=f'Reward MA{window}')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('results/learning_curve.png')

# Plot success rate
plt.figure()
plt.plot(df['Success_MA'], label=f'Success Rate MA{window}')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Success Rate over Time')
plt.legend()
plt.grid(True)
plt.savefig('results/success_rate.png')