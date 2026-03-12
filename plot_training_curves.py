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

# NeurIPS single-column width in inches
COLUMN_WIDTH = 3.5


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_tb_scalars(log_dir: str, tags: list[str]) -> pd.DataFrame:
    """
    Read scalar tags from a TensorBoard event file and return a DataFrame.

    Args:
        log_dir: Path to a lightning_logs/version_X directory.
        tags:    List of scalar tag names to extract (e.g. ["val/CER"]).

    Returns:
        DataFrame with columns [tag, step, epoch, value].
    """
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
    """
    Load TensorBoard scalars for multiple models.

    Args:
        models: {model_name: path_to_version_dir}
        tags:   scalar tags to extract

    Returns:
        DataFrame with columns [model, tag, step, value].
    """
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
# Plotting helpers
# ---------------------------------------------------------------------------

# Colour-blind friendly palette (Wong 2011)
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

SPLIT_STYLE = {"train": "-", "val": "--"}


def _get_color(model_name: str, model_names: list[str]) -> str:
    return PALETTE[model_names.index(model_name) % len(PALETTE)]


def plot_cer(
    df: pd.DataFrame,
    train_tag: str = "train/CER",
    val_tag: str = "val/CER",
    out_path: str = "cer_curves.pdf",
    x_label: str = "Step",
    smooth_weight: float = 0.0,
) -> None:
    """Plot CER curves for all models and save as publication-ready PDF + PNG."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.75))

    model_names = df["model"].unique().tolist()
    for model in model_names:
        color = _get_color(model, model_names)
        for tag, linestyle in [(train_tag, "-"), (val_tag, "--")]:
            subset = df[(df["model"] == model) & (df["tag"] == tag)].sort_values("step")
            if subset.empty:
                continue
            x = subset["step"].values
            y = _smooth(subset["value"].values, smooth_weight)
            split = tag.split("/")[0]
            label = f"{model} ({split})"
            ax.plot(x, y, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Character Error Rate (CER)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    _save(fig, out_path)


def plot_loss(
    df: pd.DataFrame,
    train_tag: str = "train/loss",
    val_tag: str = "val/loss",
    out_path: str = "loss_curves.pdf",
    x_label: str = "Step",
    smooth_weight: float = 0.0,
) -> None:
    """Plot CTC loss curves for all models and save as publication-ready PDF + PNG."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.75))

    model_names = df["model"].unique().tolist()
    for model in model_names:
        color = _get_color(model, model_names)
        for tag, linestyle in [(train_tag, "-"), (val_tag, "--")]:
            subset = df[(df["model"] == model) & (df["tag"] == tag)].sort_values("step")
            if subset.empty:
                continue
            x = subset["step"].values
            y = _smooth(subset["value"].values, smooth_weight)
            split = tag.split("/")[0]
            label = f"{model} ({split})"
            ax.plot(x, y, color=color, linestyle=linestyle, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("CTC Loss")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    _save(fig, out_path)


def _smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """Exponential moving average smoothing (weight=0 means no smoothing)."""
    if weight == 0.0:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    last = values[0]
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        smoothed[i] = last
    return smoothed


def _save(fig: plt.Figure, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    # Also save PNG alongside the PDF
    png_path = str(out_path).replace(".pdf", ".png")
    fig.savefig(png_path)
    print(f"Saved → {out_path}")
    print(f"Saved → {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main — edit the dict below to add more models
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LOG_BASE = Path("~/emg2qwerty/logs").expanduser()

    # Map model name → path to its lightning_logs/version_X directory
    MODELS = {
        "TDS-Conv-CTC": LOG_BASE / "2026-03-07/18-30-08/lightning_logs/version_0",
        # "BiLSTM":       LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
        # "BiGRU":        LOG_BASE / "YYYY-MM-DD/HH-MM-SS/lightning_logs/version_0",
    }

    # Tags logged by PyTorch Lightning for this project
    TAGS = ["train/CER", "val/CER", "train/loss", "val/loss"]

    df = load_model_logs(MODELS, TAGS)

    save_csv(df, "training_logs.csv")
    plot_cer(df, out_path="cer_curves.pdf")
    plot_loss(df, out_path="loss_curves.pdf")
