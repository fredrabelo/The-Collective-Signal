"""Recompute Table 5 (anchor-profile ratings) and the winner-selection
bootstrap directly from the raw synthetic responses and the published human
benchmark — no platform code involved.

The synthetic side is fully reproducible from data/ alone: raw per-respondent
ratings for both models are in anchor_profile_responses.csv. The human side
is NOT fully reproducible from this repository, because only the aggregate
mean/sd/n per profile are published (data/anchor_profile_human_benchmark.json)
— the original per-respondent ratings live in the Kreps et al. Harvard
Dataverse microdata (DOI 10.7910/DVN/6BSJYP), which we do not redistribute.

Two modes for the human-side bootstrap:
  1. If you place the original .dta at KREPS_DTA_PATH below (download it
     yourself from Dataverse), this script does a real respondent-level
     bootstrap on the raw human ratings, matching each anchor profile by its
     exact attribute combination.
  2. Otherwise, it falls back to a documented NORMAL APPROXIMATION: for each
     bootstrap draw, sample n_profile values from Normal(mean, sd) using the
     published aggregate statistics. This is not identical to a raw-data
     bootstrap, but was checked against it during development (raw: 65.8%
     for the 99th percentile vs. this script's normal approximation and the
     paper's published 65.3% — all within ~1 point of each other).

Usage: python3 compute_anchor_stats.py [n_bootstrap]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

KREPS_DTA_PATH = Path("../../dataverse_files 1900/Kreps_etal_vax_replication_data.dta")

ANCHOR_KEYS = [
    "anchor_1st_pct_likelihood", "anchor_25th_pct_likelihood",
    "anchor_50th_pct_likelihood", "anchor_75th_pct_likelihood", "anchor_99th_pct_likelihood",
]


def load_synthetic():
    """model -> anchor_key -> list of int ratings (1-7)."""
    out = {"claude-haiku-4-5": {k: [] for k in ANCHOR_KEYS}, "gemini-3.7-flash": {k: [] for k in ANCHOR_KEYS}}
    for row in csv.DictReader(open("../data/anchor_profile_responses.csv")):
        if row["anchor_item"] not in ANCHOR_KEYS:
            continue
        out[row["model"]][row["anchor_item"]].append(int(row["value"]))
    return out


def load_human_aggregate():
    d = json.load(open("../data/anchor_profile_human_benchmark.json"))
    return d["questions"]


def try_load_human_raw():
    """Returns anchor_key -> np.array of raw ratings, or None if the .dta
    isn't available locally (it is never redistributed by this repo)."""
    if not KREPS_DTA_PATH.exists():
        return None
    import pandas as pd

    from parse_vignette import parse_single_profile

    items = json.load(open("../data/anchor_profile_instrument.json"))
    df = pd.read_stata(KREPS_DTA_PATH, convert_categoricals=False)
    code = {
        "efficacy": {50: 1, 70: 2, 90: 3}, "duration": {1: 1, 5: 2},
        "major_side_effect": {"1/10k": 1, "1/1m": 2}, "minor_side_effect": {"1/10": 1, "1/30": 2},
        "fda_status": {"full": 1, "emergency": 2}, "origin": {"US": 1, "UK": 2, "China": 3},
        "endorsement": {"Trump": 1, "Biden": 2, "CDC": 3, "WHO": 4},
    }
    colmap = {"efficacy": "efficacy_n", "duration": "duration_n", "major_side_effect": "majorside_n",
              "minor_side_effect": "minorside_n", "fda_status": "fda_n", "origin": "origin_n", "endorsement": "endorsed_n"}
    raw = {}
    for item in items:
        attrs = parse_single_profile(item["question_text"])
        mask = np.ones(len(df), dtype=bool)
        for attr, val in attrs.items():
            mask &= (df[colmap[attr]] == code[attr][val]).values
        raw[item["synthetic_question_key"]] = df["vaxord"].values[mask]
        raw[item["synthetic_question_key"]] = raw[item["synthetic_question_key"]][~np.isnan(raw[item["synthetic_question_key"]])]
    return raw


def bootstrap_winner_frequency(rating_source, n_boot, seed=42, normal_approx_stats=None):
    """rating_source: dict anchor_key -> array of raw ratings (bootstrap by
    resampling rows), OR if normal_approx_stats is given, rating_source is
    ignored and each draw instead samples n_profile ~ Normal(mean, sd)."""
    rng = np.random.default_rng(seed)
    wins = {k: 0 for k in ANCHOR_KEYS}
    for _ in range(n_boot):
        means = {}
        for k in ANCHOR_KEYS:
            if normal_approx_stats is not None:
                stats = normal_approx_stats[k]
                sample = rng.normal(stats["mean"], stats["sd"], size=stats["n"])
            else:
                vals = rating_source[k]
                sample = rng.choice(vals, size=len(vals), replace=True)
            means[k] = sample.mean()
        wins[max(means, key=means.get)] += 1
    return {k: 100 * wins[k] / n_boot for k in ANCHOR_KEYS}


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    synthetic = load_synthetic()
    human_agg = load_human_aggregate()

    print("Table 5: mean rating, MAE, correlation, range\n")
    human_means = np.array([human_agg[k]["mean"] for k in ANCHOR_KEYS])
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        synth_means = np.array([np.mean(synthetic[model][k]) for k in ANCHOR_KEYS])
        mae = float(np.mean(np.abs(human_means - synth_means)))
        corr = float(np.corrcoef(human_means, synth_means)[0, 1])
        rng_ = synth_means.max() - synth_means.min()
        print(f"  {model}: means={[round(m,2) for m in synth_means]} MAE={mae:.2f} corr={corr:.3f} range={rng_:.2f}")
    print(f"  human: means={[round(m,2) for m in human_means]} range={human_means.max()-human_means.min():.2f}")

    print(f"\nSynthetic winner-selection bootstrap (n_boot={n_boot}, real respondent-level resampling):\n")
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        freqs = bootstrap_winner_frequency(synthetic[model], n_boot)
        print(f"  {model}: " + ", ".join(f"{k.replace('anchor_','').replace('_likelihood','')}={v:.1f}%" for k, v in freqs.items()))

    print("\nHuman winner-selection bootstrap:\n")
    raw_human = try_load_human_raw()
    if raw_human is not None:
        print(f"  Using raw Dataverse microdata found at {KREPS_DTA_PATH} (real respondent-level bootstrap):")
        freqs = bootstrap_winner_frequency(raw_human, n_boot)
    else:
        print(f"  {KREPS_DTA_PATH} not found — falling back to normal approximation from published mean/sd/n.")
        print("  Download the .dta from Harvard Dataverse (DOI 10.7910/DVN/6BSJYP) and place it at that path")
        print("  for a real respondent-level bootstrap instead.")
        freqs = bootstrap_winner_frequency(None, n_boot, normal_approx_stats=human_agg)
    print("  " + ", ".join(f"{k.replace('anchor_','').replace('_likelihood','')}={v:.1f}%" for k, v in freqs.items()))


if __name__ == "__main__":
    main()
