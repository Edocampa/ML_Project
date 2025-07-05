import pandas as pd
import matplotlib.pyplot as plt

# Carica episode metrics
df = pd.read_csv('results/dqn_results.csv')

# Compute rolling averages
window = 50
for col in ['Reward', 'Success', 'Length', 'Collisions', 'Fires']:
    df[f'{col}_MA'] = df[col].rolling(window).mean()

# Plot learning curve (reward)
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

# Plot episode length
plt.figure()
plt.plot(df['Length_MA'], label=f'Episode Length MA{window}')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode Length over Time')
plt.legend()
plt.grid(True)
plt.savefig('results/length_curve.png')

# Plot collisions and fires
plt.figure()
plt.plot(df['Collisions_MA'], label=f'Collisions MA{window}')
plt.plot(df['Fires_MA'], label=f'Fires MA{window}')
plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('Collisions & Fires over Time')
plt.legend()
plt.grid(True)
plt.savefig('results/collisions_fires.png')

# (Opzionale) loss & epsilon
df2 = pd.read_csv('results/loss_eps.csv')
# rolling mean su loss per 100 batch
df2['Loss_MA'] = df2['Loss'].rolling(100).mean()

plt.figure()
plt.plot(df2['Loss_MA'], label='Loss MA100')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_curve.png')

plt.figure()
plt.plot(df2['Epsilon'], label='Epsilon')
plt.xlabel('Batch')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay over Batches')
plt.legend()
plt.grid(True)
plt.savefig('results/epsilon_curve.png')