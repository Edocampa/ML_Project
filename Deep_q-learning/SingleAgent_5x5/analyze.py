from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR   = Path('results')
SUMMARY_CSV   = RESULTS_DIR / 'summary_all.csv'
FIG_COMP      = (10, 5)
FIG_SCENARIO  = (8, 4)
LINE_KW       = dict(linewidth=2, alpha=0.7)

plt.style.use('seaborn-v0_8-darkgrid')

# Cumulative mean function
cummean = lambda s: s.expanding(min_periods=1).mean()

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True)

if not SUMMARY_CSV.exists():
    sys.exit("summary_all.csv non trovato. Esegui prima train_experiments.py")

df = pd.read_csv(SUMMARY_CSV)

# Cumulative Reward comparison
plt.figure(figsize=FIG_COMP)
for label, grp in df.groupby('Label'):
    y = cummean(grp['Reward'])
    plt.plot(grp['Episode'], y, label=label, **LINE_KW)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Average Reward')
plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02,1))
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig(RESULTS_DIR/'comp_reward.png')
plt.close()

# Cumulative Success comparison
plt.figure(figsize=FIG_COMP)
for label, grp in df.groupby('Label'):
    y = cummean(grp['Success'] * 100)
    plt.plot(grp['Episode'], y, label=label, **LINE_KW)
plt.xlabel('Episode')
plt.ylabel('Success % (Cumulative)')
plt.title('Cumulative Success Rate')
plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02,1))
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig(RESULTS_DIR/'comp_success.png')
plt.close()

# Final success barplot remains unchanged
final_succ = df.groupby('Label')['Success'].mean() * 100
final_succ.sort_values(ascending=False).plot(kind='bar', rot=0, figsize=(6,4))
plt.ylabel('Success %')
plt.title('Final Success Rate per Scenario')
plt.tight_layout()
plt.savefig(RESULTS_DIR/'final_success.png')
plt.close()

# Per-scenario detailed plots
for scenario_dir in RESULTS_DIR.iterdir():
    if not scenario_dir.is_dir():
        continue
    label = scenario_dir.name
    ep_csv = scenario_dir / 'episode_metrics.csv'
    if not ep_csv.exists():
        continue

    plots_dir = scenario_dir / 'plots'
    ensure_dir(plots_dir)
    df_ep = pd.read_csv(ep_csv)
    if 'Episode' not in df_ep.columns:
        df_ep.insert(0, 'Episode', range(1, len(df_ep)+1))

    # Compute cumulative metrics
    for col in ['Reward', 'Success', 'Length', 'Collisions', 'Fires']:
        df_ep[f'{col}_CM'] = cummean(df_ep[col])

    # Reward CM
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Reward_CM'], **LINE_KW)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{label}: Cumulative Reward')
    plt.tight_layout()
    plt.savefig(plots_dir/'reward_cum.png')
    plt.close()

    # Success CM
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Success_CM'] * 100, **LINE_KW)
    plt.xlabel('Episode')
    plt.ylabel('Success % (Cumulative)')
    plt.title(f'{label}: Cumulative Success Rate')
    plt.tight_layout()
    plt.savefig(plots_dir/'success_cum.png')
    plt.close()

    # Length CM
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Length_CM'], **LINE_KW)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (Cumulative)')
    plt.title(f'{label}: Cumulative Episode Length')
    plt.tight_layout()
    plt.savefig(plots_dir/'length_cum.png')
    plt.close()

    # Collisions & Fires CM
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Collisions_CM'], label='Collisions', **LINE_KW)
    plt.plot(df_ep['Episode'], df_ep['Fires_CM'], label='Fires', **LINE_KW)
    plt.xlabel('Episode')
    plt.ylabel('Count (Cumulative)')
    plt.title(f'{label}: Cumulative Collisions & Fires')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(plots_dir/'collisions_fires_cum.png')
    plt.close()

    # Loss & Epsilon plots unchanged
    loss_csv = scenario_dir / 'loss_eps.csv'
    if loss_csv.exists():
        df_loss = pd.read_csv(loss_csv)
        df_loss['Loss_MA'] = df_loss['Loss'].rolling(100, min_periods=1).mean()

        # Loss MA
        plt.figure(figsize=FIG_SCENARIO)
        plt.plot(df_loss['Loss_MA'], **LINE_KW)
        plt.xlabel('Batch')
        plt.ylabel('Loss (MA100)')
        plt.title(f'{label}: Training Loss')
        plt.tight_layout()
        plt.savefig(plots_dir/'loss_ma.png')
        plt.close()

        # Epsilon decay
        plt.figure(figsize=FIG_SCENARIO)
        plt.plot(df_loss['Epsilon'], linewidth=1.5)
        plt.xlabel('Batch')
        plt.ylabel('Epsilon')
        plt.title(f'{label}: Epsilon Decay')
        plt.tight_layout()
        plt.savefig(plots_dir/'epsilon.png')
        plt.close()
