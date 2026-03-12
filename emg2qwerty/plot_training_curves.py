"""
Training curve extractor and NeurIPS-ready plotter.

Usage (script):
    python plot_training_curves.py

Usage (notebook):
    from plot_training_curves import extract_tb_scalars, plot_cer, plot_loss
    # Then call functions directly as shown at the bottom of this file.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# NeurIPS style defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COLUMN_WIDTH = 3.5  # NeurIPS single-column width in inches


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_tb_scalars(log_dir: str, tags: list[str]) -> pd.DataFrame:
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    records = []
    for tag in tags:
        if tag not in available:
            print(f"  [warn] tag '{tag}' not found in {log_dir}. Available: {sorted(available)}")
            continue
        for event in ea.Scalars(tag):
            records.append({"tag": tag, "step": event.step, "value": event.value})
    return pd.DataFrame(records, columns=["tag", "step", "value"])


def load_model_logs(models: dict[str, str], tags: list[str]) -> pd.DataFrame:
    frames = []
    for name, log_dir in models.items():
        print(f"Loading {name} from {log_dir} ...")
        df = extract_tb_scalars(log_dir, tags)
        if df.empty:
            print(f"  [warn] No data found for {name}.")
            continue
        df.insert(0, "model", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    pivot = df.pivot_table(index=["model", "step"], columns="tag", values="value").reset_index()
    pivot.columns.name = None
    pivot.to_csv(out_path, index=False)
    print(f"Saved CSV → {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
]

def _get_color(model_name, model_names):
    return PALETTE[model_names.index(model_name) % len(PALETTE)]

def _smooth(values: np.ndarray, weight: float) -> np.ndarray:
    if weight == 0.0:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    last = values[0]
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        smoothed[i] = last
    return smoothed

def _save(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    png_path = str(out_path).replace(".pdf", ".png")
    fig.savefig(png_path)
    print(f"Saved → {out_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


def plot_cer(df, train_tag="train/CER", val_tag="val/CER",
             out_path="cer_curves.pdf", x_label="Step", smooth_weight=0.0):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.75))
    model_names = df["model"].unique().tolist()
    for model in model_names:
        color = _get_color(model, model_names)
        for tag, ls in [(train_tag, "-"), (val_tag, "--")]:
            subset = df[(df["model"] == model) & (df["tag"] == tag)].sort_values("step")
            if subset.empty:
                continue
            ax.plot(subset["step"].values, _smooth(subset["value"].values, smooth_weight),
                    color=color, linestyle=ls, label=f"{model} ({tag.split('/')[0]})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Character Error Rate (CER)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    _save(fig, out_path)


def plot_loss(df, train_tag="train/loss", val_tag="val/loss",
              out_path="loss_curves.pdf", x_label="Step", smooth_weight=0.0):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.75))
    model_names = df["model"].unique().tolist()
    for model in model_names:
        color = _get_color(model, model_names)
        for tag, ls in [(train_tag, "-"), (val_tag, "--")]:
            subset = df[(df["model"] == model) & (df["tag"] == tag)].sort_values("step")
            if subset.empty:
                continue
            ax.plot(subset["step"].values, _smooth(subset["value"].values, smooth_weight),
                    color=color, linestyle=ls, label=f"{model} ({tag.split('/')[0]})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("CTC Loss")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main — edit MODELS dict to add BiLSTM, BiGRU, etc.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LOG_BASE = Path("~/emg2qwerty/logs").expanduser()

    MODELS = {
    #"Baseline":           LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
    #"CNN+RNN":            LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
    "BiLSTM":             LOG_BASE / "2026-03-07/18-30-08/lightning_logs/version_0",
    #"BiGRU":              LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
    #"Attention":          LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
    #"BiLSTM + Attention": LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
}

    TAGS = ["train/CER", "val/CER", "train/loss", "val/loss"]

    df = load_model_logs(MODELS, TAGS)
    save_csv(df, "training_logs.csv")
    plot_cer(df, out_path="cer_curves.pdf")
    plot_loss(df, out_path="loss_curves.pdf")