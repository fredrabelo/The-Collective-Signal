"""Recompute the shared 24-pair battery's synthetic AMCEs (paper Table 3 /
RQ2) directly from the raw responses and the literal instrument text — no
platform code involved. The paper's primary estimator is OLS on all seven
attributes simultaneously (adopted because the 24 pairs are a fixed,
non-fully-randomized battery in which attribute levels can be mildly
correlated, so a single-attribute difference-in-means does not fully adjust
for the other six). This script also reports the simple difference-in-means
AMCE as a sensitivity check alongside it, on the same synthetic choices —
this is descriptive, not a formal statistical bound on anything; it just
shows how much the estimator choice itself moves the numbers for this
specific battery. (Human-side AMCEs are not recomputed here — that requires
the original Kreps et al. Stata microdata, not redistributed in this
package; see data/README.md. The published human values are reproduced from
the paper for reference only.)

Usage: python3 compute_battery_amce.py
"""
import csv
import json
from collections import defaultdict

import numpy as np

from parse_vignette import ATTR_ORDER, REFERENCE, parse_pair

HUMAN_AMCE = {  # from paper Table 3, for reference/comparison only
    ("efficacy", 70): 7.35, ("efficacy", 90): 16.58,
    ("duration", 5): 5.41,
    ("major_side_effect", "1/1m"): 6.58,
    ("minor_side_effect", "1/30"): 1.39,
    ("fda_status", "emergency"): -2.99,
    ("origin", "UK"): -3.62, ("origin", "China"): -13.23,
    ("endorsement", "Biden"): 2.02, ("endorsement", "CDC"): 9.20, ("endorsement", "WHO"): 7.75,
}

CONTRASTS = list(HUMAN_AMCE.keys())


def load_pairs():
    items = json.load(open("../data/shared_battery_instrument.json"))
    return {item["synthetic_question_key"]: parse_pair(item["question_text"]) for item in items}


VALID_PAIR_KEYS = {"pair%02d_choice" % n for n in range(1, 25)}


def load_choices(model):
    """Returns pair_key -> [(external_key, chosen_side), ...], skipping any
    row whose key isn't one of the 24 real pairs (see data/README.md for the
    two known malformed rows from a single respondent's Claude batch reply)."""
    by_pair = defaultdict(list)
    skipped = 0
    with open("../data/shared_battery_responses.csv") as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            if row["pair"] not in VALID_PAIR_KEYS:
                skipped += 1
                continue
            by_pair[row["pair"]].append((row["external_key"], row["value"]))
    if skipped:
        print(f"  ({model}: skipped {skipped} row(s) with non-standard keys — see data/README.md)")
    return by_pair


def build_long_table(pairs, choices):
    """One row per (respondent, pair, side): attribute levels + whether that
    side was chosen. Feeds both the simple-difference and the OLS estimator."""
    rows = []
    for pair_key, (attr_a, attr_b) in pairs.items():
        for ext_key, choice in choices.get(pair_key, []):
            for side_label, attrs in (("Vaccine A", attr_a), ("Vaccine B", attr_b)):
                rows.append({**attrs, "chosen": 1 if choice == side_label else 0})
    return rows


def build_respondent_blocks(pairs, choices):
    """Same rows as build_long_table, but grouped by respondent, so a
    bootstrap can resample respondents (not individual rows) with
    replacement — respecting that each respondent contributes 48 correlated
    profile-side observations (24 pairs x 2 sides)."""
    blocks = defaultdict(list)
    for pair_key, (attr_a, attr_b) in pairs.items():
        for ext_key, choice in choices.get(pair_key, []):
            for side_label, attrs in (("Vaccine A", attr_a), ("Vaccine B", attr_b)):
                blocks[ext_key].append({**attrs, "chosen": 1 if choice == side_label else 0})
    return blocks


def simple_diff_amce(rows):
    """AMCE as mean(chosen | level) - mean(chosen | reference level), per
    attribute — the same estimator the paper's Method section describes."""
    out = {}
    for attr in ATTR_ORDER:
        by_level = defaultdict(list)
        for r in rows:
            by_level[r[attr]].append(r["chosen"])
        ref_mean = 100 * np.mean(by_level[REFERENCE[attr]])
        for level, vals in by_level.items():
            if level == REFERENCE[attr]:
                continue
            out[(attr, level)] = 100 * np.mean(vals) - ref_mean
    return out


def ols_adjusted_amce(rows):
    """Same outcome, but regression-adjusted for all seven attributes at
    once (one-hot, reference level omitted) — checks whether the fixed
    battery's attribute correlation structure is inflating/distorting the
    simple-difference estimates above."""
    contrast_index = {c: i for i, c in enumerate(CONTRASTS)}
    X = np.zeros((len(rows), len(CONTRASTS) + 1))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        X[i, 0] = 1.0
        for attr in ATTR_ORDER:
            level = r[attr]
            if level != REFERENCE[attr] and (attr, level) in contrast_index:
                X[i, 1 + contrast_index[(attr, level)]] = 1.0
        y[i] = r["chosen"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {c: 100 * coef[1 + i] for c, i in contrast_index.items()}


def summarize(amce, label):
    human = np.array([HUMAN_AMCE[c] for c in CONTRASTS])
    synth = np.array([amce[c] for c in CONTRASTS])
    directions = int(np.sum(np.sign(human) == np.sign(synth)))
    mae = float(np.mean(np.abs(human - synth)))
    corr = float(np.corrcoef(human, synth)[0, 1])
    print(f"\n{label}: direction agreement {directions}/11, MAE {mae:.2f}pp, correlation {corr:.3f}")
    for c in CONTRASTS:
        print(f"    {c[0]:<20}{str(c[1]):<8} human {HUMAN_AMCE[c]:+7.2f}  synth {amce[c]:+7.2f}")


def stats_from_amce(amce):
    human = np.array([HUMAN_AMCE[c] for c in CONTRASTS])
    synth = np.array([amce[c] for c in CONTRASTS])
    directions = int(np.sum(np.sign(human) == np.sign(synth)))
    mae = float(np.mean(np.abs(human - synth)))
    corr = float(np.corrcoef(human, synth)[0, 1])
    return directions, mae, corr


def ci(values, lo=2.5, hi=97.5):
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def bootstrap(blocks, n_boot, seed=42):
    """Respondent-clustered bootstrap: resample respondents with
    replacement, refit OLS on each resample's full row set."""
    keys = list(blocks.keys())
    rng = np.random.default_rng(seed)
    n = len(keys)
    boot_amce = {c: [] for c in CONTRASTS}
    boot_directions, boot_mae, boot_corr = [], [], []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        rows = []
        for i in sample_idx:
            rows.extend(blocks[keys[i]])
        amce = ols_adjusted_amce(rows)
        for c in CONTRASTS:
            boot_amce[c].append(amce[c])
        d, m, r = stats_from_amce(amce)
        boot_directions.append(d)
        boot_mae.append(m)
        boot_corr.append(r)
    return boot_amce, boot_directions, boot_mae, boot_corr


def main():
    import sys
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pairs = load_pairs()
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        choices = load_choices(model)
        rows = build_long_table(pairs, choices)
        print(f"\n{'='*70}\n{model}: {len(rows)} profile-side observations\n{'='*70}")
        summarize(simple_diff_amce(rows), "Simple difference-in-means (sensitivity check)")
        point_amce = ols_adjusted_amce(rows)
        summarize(point_amce, "OLS, all 7 attributes adjusted simultaneously (paper's estimator)")

        if n_boot > 0:
            blocks = build_respondent_blocks(pairs, choices)
            boot_amce, boot_d, boot_mae, boot_corr = bootstrap(blocks, n_boot)
            d_lo, d_hi = ci(boot_d)
            mae_lo, mae_hi = ci(boot_mae)
            corr_lo, corr_hi = ci(boot_corr)
            print(f"\n  Respondent-clustered bootstrap (n_boot={n_boot}): "
                  f"directions [{d_lo:.1f},{d_hi:.1f}], "
                  f"MAE [{mae_lo:.2f},{mae_hi:.2f}], "
                  f"corr [{corr_lo:.3f},{corr_hi:.3f}]")
            for c in CONTRASTS:
                lo, hi = ci(boot_amce[c])
                print(f"    {c[0]:<20}{str(c[1]):<8} {point_amce[c]:>+7.2f}  [{lo:>+6.2f}, {hi:>+6.2f}]")


if __name__ == "__main__":
    main()
