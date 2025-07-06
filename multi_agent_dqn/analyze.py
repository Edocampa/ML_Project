from pathlib import Path
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path('results')
SUMMARY_CSV = RESULTS_DIR / 'summary_all.csv'
MA_WINDOW   = 50


# Utilità
rolling = lambda s: s.rolling(MA_WINDOW, min_periods=1).mean()

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True)


df = pd.read_csv(SUMMARY_CSV)

# Reward MA comparison
plt.figure(figsize=(7,4))
for label, grp in df.groupby('Label'):
    plt.plot(grp['Episode'], rolling(grp['Reward']), label=label)
plt.xlabel('Episode'); plt.ylabel('Reward (MA)')
plt.title(f'Average Reward – Window {MA_WINDOW}')
plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(RESULTS_DIR/'comp_reward.png'); plt.close()

# Success MA comparison
plt.figure(figsize=(7,4))
for label, grp in df.groupby('Label'):
    plt.plot(grp['Episode'], rolling(grp['Success']*100), label=label)
plt.xlabel('Episode'); plt.ylabel('Success % (MA)')
plt.title(f'Success Rate – Window {MA_WINDOW}')
plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(RESULTS_DIR/'comp_success.png'); plt.close()

# Final success barplot
final_succ = df.groupby('Label')['Success'].mean()*100
final_succ.sort_values(ascending=False).plot(kind='bar', rot=0)
plt.ylabel('Success %'); plt.title('Final Success Rate per Scenario')
plt.tight_layout(); plt.savefig(RESULTS_DIR/'final_success.png'); plt.close()


for scenario_dir in RESULTS_DIR.iterdir():
    if not scenario_dir.is_dir():
        continue
    label = scenario_dir.name
    ep_path = scenario_dir / 'episode_metrics.csv'
    if not ep_path.exists():
        continue

    plots_dir = scenario_dir / 'plots'
    ensure_dir(plots_dir)
    df_ep = pd.read_csv(ep_path)

    # Moving averages
    for col in ['Reward','Success','Length','Collisions','Fires']:
        df_ep[f'{col}_MA'] = rolling(df_ep[col])

    # Reward MA
    plt.figure(); plt.plot(df_ep['Reward_MA'])
    plt.xlabel('Episode'); plt.ylabel('Average Reward')
    plt.title(f'{label}: Reward (MA{MA_WINDOW})'); plt.grid(True)
    plt.tight_layout(); plt.savefig(plots_dir/'reward_ma.png'); plt.close()

    # Success rate MA
    plt.figure(); plt.plot(df_ep['Success_MA']*100)
    plt.xlabel('Episode'); plt.ylabel('Success %')
    plt.title(f'{label}: Success Rate (MA{MA_WINDOW})'); plt.grid(True)
    plt.tight_layout(); plt.savefig(plots_dir/'success_ma.png'); plt.close()

    # Episode length MA
    plt.figure(); plt.plot(df_ep['Length_MA'])
    plt.xlabel('Episode'); plt.ylabel('Length')
    plt.title(f'{label}: Episode Length (MA{MA_WINDOW})'); plt.grid(True)
    plt.tight_layout(); plt.savefig(plots_dir/'length_ma.png'); plt.close()

    # Collisions & Fires MA
    plt.figure();
    plt.plot(df_ep['Collisions_MA'], label='Collisions')
    plt.plot(df_ep['Fires_MA'], label='Fires')
    plt.xlabel('Episode'); plt.ylabel('Count')
    plt.title(f'{label}: Collisions & Fires (MA{MA_WINDOW})'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(plots_dir/'collisions_fires_ma.png'); plt.close()

    # Loss & Epsilon (if available)
    loss_path = scenario_dir / 'loss_eps.csv'
    if loss_path.exists():
        df_loss = pd.read_csv(loss_path)
        df_loss['Loss_MA'] = df_loss['Loss'].rolling(100, min_periods=1).mean()

        plt.figure(); plt.plot(df_loss['Loss_MA'])
        plt.xlabel('Batch'); plt.ylabel('Loss (MA100)')
        plt.title(f'{label}: Training Loss')
        plt.grid(True); plt.tight_layout()
        plt.savefig(plots_dir/'loss_ma.png'); plt.close()

        plt.figure(); plt.plot(df_loss['Epsilon'])
        plt.xlabel('Batch'); plt.ylabel('Epsilon')
        plt.title(f'{label}: Epsilon Decay')
        plt.grid(True); plt.tight_layout()
        plt.savefig(plots_dir/'epsilon.png'); plt.close()

