"""Compares Gemini's 84 incomplete shared-battery respondents (truncated-JSON
batch failures, out of 1,971) against the 1,887 who completed it, and checks
whether restricting Claude's comparison to that same 1,887-respondent subset
changes Claude's AMCE fidelity numbers.

Usage: python3 check_gemini_missingness.py
"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from parse_vignette import ATTR_ORDER, REFERENCE, parse_pair

HUMAN_AMCE = {
    ("efficacy", 70): 7.35, ("efficacy", 90): 16.58, ("duration", 5): 5.41,
    ("major_side_effect", "1/1m"): 6.58, ("minor_side_effect", "1/30"): 1.39,
    ("fda_status", "emergency"): -2.99, ("origin", "UK"): -3.62, ("origin", "China"): -13.23,
    ("endorsement", "Biden"): 2.02, ("endorsement", "CDC"): 9.20, ("endorsement", "WHO"): 7.75,
}
CONTRASTS = list(HUMAN_AMCE.keys())
VALID_PAIR_KEYS = {"pair%02d_choice" % n for n in range(1, 25)}


def covariate_comparison():
    gemini_respondents, claude_respondents = set(), set()
    for row in csv.DictReader(open("../data/shared_battery_responses.csv")):
        if row["pair"] not in VALID_PAIR_KEYS:
            continue
        if row["model"] == "gemini-3.7-flash":
            gemini_respondents.add(row["external_key"])
        elif row["model"] == "claude-haiku-4-5":
            claude_respondents.add(row["external_key"])
    missing = claude_respondents - gemini_respondents
    print(f"Complete (both models): {len(gemini_respondents)}; "
          f"missing from Gemini only: {len(missing)}\n")

    basic = json.load(open("../data/respondents_basic.json"))
    attrs = {r["external_key"]: {e["evidence_key"]: e["value"] for e in r["evidence"]} for r in basic}

    def dist(keys, field):
        c = Counter(attrs[k][field] for k in keys if k in attrs)
        total = sum(c.values())
        return {k: round(100 * v / total, 1) for k, v in c.items()}

    for field in ("party", "gender", "race", "education"):
        print(f"{field}:")
        print("  complete:", dist(gemini_respondents, field))
        print("  missing :", dist(missing, field))

    ages_complete = [attrs[k]["age"] for k in gemini_respondents if k in attrs]
    ages_missing = [attrs[k]["age"] for k in missing if k in attrs]
    print(f"\nage: complete mean={np.mean(ages_complete):.1f} sd={np.std(ages_complete, ddof=1):.1f}  "
          f"missing mean={np.mean(ages_missing):.1f} sd={np.std(ages_missing, ddof=1):.1f}")
    return gemini_respondents, claude_respondents


def ols_amce(rows):
    idx = {c: i for i, c in enumerate(CONTRASTS)}
    X = np.zeros((len(rows), len(CONTRASTS) + 1))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        X[i, 0] = 1.0
        for attr in ATTR_ORDER:
            level = r[attr]
            if level != REFERENCE[attr] and (attr, level) in idx:
                X[i, 1 + idx[(attr, level)]] = 1.0
        y[i] = r["chosen"]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {c: 100 * coef[1 + i] for c, i in idx.items()}


def stats(amce):
    human = np.array([HUMAN_AMCE[c] for c in CONTRASTS])
    synth = np.array([amce[c] for c in CONTRASTS])
    d = int(np.sum(np.sign(human) == np.sign(synth)))
    mae = float(np.mean(np.abs(human - synth)))
    corr = float(np.corrcoef(human, synth)[0, 1])
    return d, mae, corr


def sensitivity_check(gemini_respondents):
    items = json.load(open("../data/shared_battery_instrument.json"))
    pairs = {item["synthetic_question_key"]: parse_pair(item["question_text"]) for item in items}
    claude_rows = defaultdict(list)
    for row in csv.DictReader(open("../data/shared_battery_responses.csv")):
        if row["model"] == "claude-haiku-4-5" and row["pair"] in VALID_PAIR_KEYS:
            claude_rows[row["external_key"]].append((row["pair"], row["value"]))

    def build_rows(respondent_filter):
        rows = []
        for ext, entries in claude_rows.items():
            if ext not in respondent_filter:
                continue
            for pair_key, choice in entries:
                a, b = pairs[pair_key]
                rows.append({**a, "chosen": 1 if choice == "Vaccine A" else 0})
                rows.append({**b, "chosen": 1 if choice == "Vaccine B" else 0})
        return rows

    print("\nSensitivity check: does restricting Claude to Gemini's completers change anything?\n")
    d, mae, corr = stats(ols_amce(build_rows(set(claude_rows.keys()))))
    print(f"  Claude, full N=1,971:                     dir {d}/11  MAE {mae:.2f}  corr {corr:.3f}")
    d, mae, corr = stats(ols_amce(build_rows(gemini_respondents)))
    print(f"  Claude, restricted to Gemini's N={len(gemini_respondents)}: dir {d}/11  MAE {mae:.2f}  corr {corr:.3f}")


if __name__ == "__main__":
    gemini_respondents, _ = covariate_comparison()
    sensitivity_check(gemini_respondents)
