"""Generate the four manuscript figures from the paper's published values.

Every value plotted here matches what `compute_matched_replay_amce.py`,
`compute_battery_amce.py`, `compute_partisan_gaps.py`, and
`compute_anchor_stats.py` compute directly from the raw data in `../data/`
— run those scripts to verify any number independently. Both PDF (vector,
for LaTeX) and PNG (quick preview) versions are produced.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"Human": "#333333", "Claude": "#0072B2", "Gemini": "#D55E00"}
MARKERS = {"Human": "o", "Claude": "s", "Gemini": "^"}


def style(ax):
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def amce_figure():
    labels = [
        "Efficacy: 70% vs 50%", "Efficacy: 90% vs 50%",
        "Protection: 5 years vs 1", "Major side effect: 1/1m vs 1/10k",
        "Minor side effect: 1/30 vs 1/10", "FDA: emergency vs full",
        "Origin: U.K. vs U.S.", "Origin: China vs U.S.",
        "Endorsement: Biden vs Trump", "Endorsement: CDC vs Trump",
        "Endorsement: WHO vs Trump",
    ]
    values = {
        "Human": [7.35, 16.58, 5.41, 6.58, 1.39, -2.99, -3.62, -13.23, 2.02, 9.20, 7.75],
        "Claude": [37.50, 69.29, 4.34, -8.66, 4.01, 3.26, -9.79, -16.54, 26.90, 11.26, 22.56],
        "Gemini": [29.16, 64.75, 10.15, -9.87, 0.18, -5.18, -6.89, -21.33, 28.27, 17.53, 29.62],
    }
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.25, 5.1), constrained_layout=True)
    ax.axvline(0, color="#777777", linewidth=0.9)
    # One horizontal guide per contrast makes the comparison unit explicit.
    # All three sources sit at exactly the same y coordinate.
    for row in y:
        ax.axhline(row, color="#e1e1e1", linewidth=0.75, zorder=0)
    for name in ["Human", "Claude", "Gemini"]:
        ax.scatter(values[name], y, s=38, marker=MARKERS[name],
                   color=COLORS[name], edgecolor="white", linewidth=0.45,
                   label=name, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Average marginal component effect (percentage points)", fontsize=9.5)
    ax.set_title("Attribute effects: directions often agree, magnitudes do not", fontsize=11, weight="bold")
    ax.legend(
        ncol=3,
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        borderaxespad=0,
    )
    style(ax)
    save(fig, "figure1_amce_fidelity")


def partisan_figure():
    labels = ["Trump", "Biden", "CDC", "WHO"]
    values = {
        "Human": [17.25, -12.01, -3.14, -5.34],
        "Claude": [36.83, -13.61, 4.64, -25.09],
        "Gemini": [32.26, -27.31, 7.17, -25.90],
    }
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
    ax.axvline(0, color="#777777", linewidth=0.9)
    for row in y:
        ax.axhline(row, color="#e1e1e1", linewidth=0.75, zorder=0)
    for name in ["Human", "Claude", "Gemini"]:
        ax.scatter(values[name], y, s=28, marker=MARKERS[name],
                   color=COLORS[name], edgecolor="white", linewidth=0.45,
                   label=name, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Republican minus Democrat choice (pp)", fontsize=8)
    ax.set_title("Partisan endorsement gaps", fontsize=9.5, weight="bold")
    ax.legend(ncol=3, frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), borderaxespad=0,
              handletextpad=0.35, columnspacing=0.8)
    style(ax)
    save(fig, "figure2_partisan_gaps")


def decision_figure():
    labels = ["1st", "25th", "50th", "75th", "99th"]
    ratings = {
        "Human": [3.79, 4.06, 4.51, 4.79, 5.06],
        "Claude": [2.45, 4.46, 4.80, 6.30, 6.92],
        "Gemini": [2.21, 3.63, 4.07, 5.46, 6.45],
    }
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    for name in ["Human", "Claude", "Gemini"]:
        ax.plot(x, ratings[name], marker=MARKERS[name], color=COLORS[name],
                linewidth=1.5, markersize=4.5, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(1, 7.15)
    ax.set_ylabel("Mean likelihood rating (1–7)", fontsize=8)
    ax.set_xlabel("Profile percentile", fontsize=8)
    ax.set_title("Ranking agrees; separation expands", fontsize=9.5, weight="bold")
    ax.grid(color="#dedede", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7.5)
    ax.set_axisbelow(True)
    save(fig, "figure3_decision_fidelity")


def matched_fidelity_figure():
    labels = ["Claude\nbasic", "Claude\nrich", "Gemini\nbasic", "Gemini\nrich"]
    raw = [50.76, 51.57, 51.39, 53.04]
    balanced = [43.20, 44.40, 44.99, 48.51]
    recall_a = [63.68, 64.59, 62.87, 61.71]
    recall_b = [63.61, 62.83, 61.59, 59.62]
    recall_neither = [2.32, 5.79, 10.53, 24.19]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)
    axes[0].bar(x - .17, raw, .34, color="#D97A57", label="Raw accuracy")
    axes[0].bar(x + .17, balanced, .34, color="#E9B9A8", label="Balanced accuracy")
    axes[0].axhline(100 / 3, color="#555555", linestyle="--", linewidth=.9,
                    label="Uniform-chance baseline")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 70)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Individual matched-choice fidelity")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].grid(axis="y", color="#dedede", linewidth=.6)
    axes[1].plot(x, recall_a, marker="o", color="#2F6B3C", label="Vaccine A")
    axes[1].plot(x, recall_b, marker="s", color="#6A4C93", label="Vaccine B")
    axes[1].plot(x, recall_neither, marker="D", color="#B54A32", label="Neither")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 70)
    axes[1].set_ylabel("Recall (percent)")
    axes[1].set_title("The richness gain is concentrated in 'Neither'")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[1].grid(axis="y", color="#dedede", linewidth=.6)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)
    save(fig, "figure4_matched_fidelity")


if __name__ == "__main__":
    amce_figure()
    partisan_figure()
    decision_figure()
    matched_fidelity_figure()
    print(f"Wrote manuscript figures to {OUT}")
