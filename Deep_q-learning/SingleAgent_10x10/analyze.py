from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
RESULTS_DIR   = Path("results")
CONFIGS       = ["A-base", "B-miniB", "C-smallRB", "D-fastE"]
SMOOTH_WINDOW = 200
FIGSIZE       = (12, 8)

kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
data = {}
for label in CONFIGS:
    ep_csv = RESULTS_DIR / label / "episode_metrics.csv"
    if not ep_csv.exists():
        continue

    df_ep = pd.read_csv(ep_csv)
    data[label] = {
        "reward":  df_ep["Reward"].values,
        "steps":   df_ep["Length"].values,
        "success": np.cumsum(df_ep["Success"].values) /
                   np.arange(1, len(df_ep) + 1),
    }

    # ---- cerca il file di loss -------------------------------------------
    loss_path_candidates = [
        RESULTS_DIR / label / "loss_eps.csv",   # <-- aggiunto
        RESULTS_DIR / label / "train_loss.csv",
        RESULTS_DIR / label / "loss.csv",
    ]
    for p in loss_path_candidates:
        if p.exists():
            df_loss = pd.read_csv(p)
            if "Loss" in df_loss.columns:
                data[label]["loss"] = df_loss["Loss"].values
            break
    else:
        data[label]["loss"] = None


metrics = [
    ("reward",  "Total Reward per Episode",  "Reward",  "Episode"),
    ("steps",   "Steps per Episode",         "Steps",   "Episode"),
    ("success", "Cumulative Success Rate",   "Success", "Episode"),
    ("loss",    "Training Loss",             "Loss",    "Episode"),   # <-- qui
]


# -----------------------------------------------------------------------------
# plotting loop
# -----------------------------------------------------------------------------
for key, title, ylabel, xlabel in metrics:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(title, fontsize=16)

    for idx, label in enumerate(CONFIGS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        y  = data[label].get(key)

        if y is None or len(y) == 0:
            ax.text(0.5, 0.5, "no loss data" if key == "loss" else "no data",
                    ha="center", va="center", fontsize=12, color="red")
            ax.set_axis_off()
        else:
            # raw trace
            ax.plot(y, color="lightgray", label="raw")

            # moving average
            if len(y) >= SMOOTH_WINDOW:
                smooth = np.convolve(y, kernel, mode="valid")
                x_smooth = np.arange(SMOOTH_WINDOW - 1, len(y))
                ax.plot(x_smooth, smooth,
                        linewidth=2, label=f"{SMOOTH_WINDOW}-step MA")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

        ax.set_title(label)

    # rimuovi pannelli extra se meno di 4
    for j in range(len(CONFIGS), 4):
        fig.delaxes(axes.flat[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = RESULTS_DIR / f"compare_{key}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

print("Plots saved in", RESULTS_DIR.resolve())
