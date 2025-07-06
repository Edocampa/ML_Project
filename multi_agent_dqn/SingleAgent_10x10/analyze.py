from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── config ──────────────────────────
RESULTS_DIR   = Path('results')          # cartella dei nuovi esperimenti
SUMMARY_CSV   = RESULTS_DIR / 'summary_all.csv'
MA_WINDOW     = 200
FIG_COMP      = (10, 5)
FIG_SCENARIO  = (8, 4)
LINE_KW       = dict(linewidth=2, alpha=0.7)

plt.style.use('seaborn-v0_8-darkgrid')
rolling = lambda s: s.rolling(MA_WINDOW, min_periods=1).mean()

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True)


df = pd.read_csv(SUMMARY_CSV)

# Reward MA comparison
plt.figure(figsize=FIG_COMP)
for label, grp in df.groupby('Label'):
    plt.plot(grp['Episode'], rolling(grp['Reward']), label=label, **LINE_KW)
plt.xlabel('Episode'); plt.ylabel('Reward (MA)')
plt.title(f'Average Reward – Window {MA_WINDOW}')
plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02,1))
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig(RESULTS_DIR/'comp_reward.png'); plt.close()

# Success MA comparison
plt.figure(figsize=FIG_COMP)
for label, grp in df.groupby('Label'):
    plt.plot(grp['Episode'], rolling(grp['Success']*100), label=label, **LINE_KW)
plt.xlabel('Episode'); plt.ylabel('Success % (MA)')
plt.title(f'Success Rate – Window {MA_WINDOW}')
plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02,1))
plt.tight_layout(rect=[0,0,0.85,1])
plt.savefig(RESULTS_DIR/'comp_success.png'); plt.close()

# Final success barplot
(final_succ := df.groupby('Label')['Success'].mean()*100)
final_succ.sort_values(ascending=False).plot(kind='bar', rot=0, figsize=(6,4))
plt.ylabel('Success %'); plt.title('Final Success Rate per Scenario')
plt.tight_layout(); plt.savefig(RESULTS_DIR/'final_success.png'); plt.close()


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

    for col in ['Reward','Success','Length','Collisions','Fires']:
        df_ep[f'{col}_MA'] = rolling(df_ep[col])

    # Reward MA
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Reward_MA'], **LINE_KW)
    plt.xlabel('Episode'); plt.ylabel('Average Reward')
    plt.title(f'{label}: Reward (MA{MA_WINDOW})')
    plt.tight_layout(); plt.savefig(plots_dir/'reward_ma.png'); plt.close()

    # Success MA
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Success_MA']*100, **LINE_KW)
    plt.xlabel('Episode'); plt.ylabel('Success %')
    plt.title(f'{label}: Success Rate (MA{MA_WINDOW})')
    plt.tight_layout(); plt.savefig(plots_dir/'success_ma.png'); plt.close()

    # Length MA
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Length_MA'], **LINE_KW)
    plt.xlabel('Episode'); plt.ylabel('Length')
    plt.title(f'{label}: Episode Length (MA{MA_WINDOW})')
    plt.tight_layout(); plt.savefig(plots_dir/'length_ma.png'); plt.close()

    # Collisions & Fires
    plt.figure(figsize=FIG_SCENARIO)
    plt.plot(df_ep['Episode'], df_ep['Collisions_MA'], label='Collisions', **LINE_KW)
    plt.plot(df_ep['Episode'], df_ep['Fires_MA'], label='Fires', **LINE_KW)
    plt.xlabel('Episode'); plt.ylabel('Count')
    plt.title(f'{label}: Collisions & Fires (MA{MA_WINDOW})')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(plots_dir/'collisions_fires_ma.png'); plt.close()

    # Loss & epsilon
    loss_csv = scenario_dir / 'loss_eps.csv'
    if loss_csv.exists():
        df_loss = pd.read_csv(loss_csv)
        df_loss['Loss_MA'] = df_loss['Loss'].rolling(100, min_periods=1).mean()

        plt.figure(figsize=FIG_SCENARIO)
        plt.plot(df_loss['Loss_MA'], **LINE_KW)
        plt.xlabel('Batch'); plt.ylabel('Loss (MA100)')
        plt.title(f'{label}: Training Loss')
        plt.tight_layout(); plt.savefig(plots_dir/'loss_ma.png'); plt.close()

        plt.figure(figsize=FIG_SCENARIO)
        plt.plot(df_loss['Epsilon'], linewidth=1.5)
        plt.xlabel('Batch'); plt.ylabel('Epsilon')
        plt.title(f'{label}: Epsilon Decay')
        plt.tight_layout(); plt.savefig(plots_dir/'epsilon.png'); plt.close()
