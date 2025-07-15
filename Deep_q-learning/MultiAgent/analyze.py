from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR  = Path('results')
MA_WINDOW    = 50
LINE_KW      = dict(linewidth=2, alpha=0.7)
plt.style.use('seaborn-v0_8-darkgrid')
rolling = lambda s: s.rolling(MA_WINDOW, min_periods=1).mean()


def plot_reward(agent_id: int):
    csv = RESULTS_DIR / f'agent{agent_id}_metrics.csv'
    if not csv.exists():
        return
    df   = pd.read_csv(csv)
    arr  = df['Reward'].values

    plt.figure(figsize=(9,4))
    plt.plot(arr, color='lightgray', label='raw')
    if len(arr) >= MA_WINDOW:
        plt.plot(rolling(df['Reward']), **LINE_KW, label=f'MA{MA_WINDOW}')
    plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.title(f'Agent {agent_id} – Reward (MA{MA_WINDOW})')
    plt.grid(True); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f'reward_agent{agent_id}.png'); plt.close()


for i in (0,1):
    plot_reward(i)


csv0 = RESULTS_DIR / 'agent0_metrics.csv'
if csv0.exists():
    succ = pd.read_csv(csv0)['Success'].values
    cum  = np.cumsum(succ) / np.arange(1, len(succ)+1) * 100
    plt.figure(figsize=(9,4))
    plt.plot(cum, **LINE_KW)
    plt.xlabel('Episode'); plt.ylabel('Success % (cumulative)')
    plt.title('Team Cumulative Success Rate')
    plt.grid(True); plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'success_cum.png'); plt.close()

