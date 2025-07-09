from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR   = Path('results')
CONFIGS       = ['A-base', 
                 #'B-miniB', 
                 #'C-smallRB', 
                 #'D-fastE'
                 ]  # tutti e quattro i casi
SMOOTH_WINDOW = 200
FIGSIZE       = (12, 8)
kernel        = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW

# Load data
data = {}
for label in CONFIGS:
    ep_csv = RESULTS_DIR/label/'episode_metrics.csv'
    if not ep_csv.exists():
        continue
    df_ep = pd.read_csv(ep_csv)

    data[label] = {
        'reward':  df_ep['Reward'].values,
        'steps':   df_ep['Length'].values,
        'success': np.cumsum(df_ep['Success'].values) / np.arange(1, len(df_ep)+1)
    }

# Define metrics to plot and their titles
metrics = [
    ('reward',  'Total Reward per Episode'),
    ('steps',   'Steps per Episode'),
    ('success', 'Cumulative Success Rate')
]

# Plot each metric in a 2x2 grid and save to file
for key, title in metrics:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=16)

    for idx, label in enumerate(CONFIGS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        y  = data[label][key]

        # raw trace
        ax.plot(y, color='lightgray', label='raw')

        # moving average
        if len(y) >= SMOOTH_WINDOW:
            smooth = np.convolve(y, kernel, mode='valid')
            x = np.arange(SMOOTH_WINDOW - 1, len(y))
            ax.plot(x, smooth, label=f'{SMOOTH_WINDOW}-ep MA', linewidth=2)

        ax.set_title(label)
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()

    # remove unused axes
    for j in range(len(CONFIGS), 4):
        fig.delaxes(axes.flat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = RESULTS_DIR/f"compare_{key}.png"
    fig.savefig(out_file)
    plt.close(fig)

print("Plots saved in", RESULTS_DIR)
