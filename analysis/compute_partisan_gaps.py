"""Recompute the shared-battery Republican-minus-Democrat choice gaps by
endorsement (paper Table 4 / RQ3) directly from raw responses + respondent
party affiliation. Independent check on whether Table 4, like Table 3,
reproduces exactly from released data.

Usage: python3 compute_partisan_gaps.py
"""
import csv
import json
from collections import defaultdict

import numpy as np

from parse_vignette import parse_pair

ENDORSEMENTS = ["Trump", "Biden", "CDC", "WHO"]
VALID_PAIR_KEYS = {"pair%02d_choice" % n for n in range(1, 25)}


def load_party():
    d = json.load(open("../data/respondents_basic.json"))
    party = {}
    for r in d:
        for e in r["evidence"]:
            if e["evidence_key"] == "party":
                party[r["external_key"]] = e["value"]
    return party


def load_pairs():
    items = json.load(open("../data/shared_battery_instrument.json"))
    return {item["synthetic_question_key"]: parse_pair(item["question_text"]) for item in items}


def load_choices(model):
    by_pair = defaultdict(list)
    for row in csv.DictReader(open("../data/shared_battery_responses.csv")):
        if row["model"] != model or row["pair"] not in VALID_PAIR_KEYS:
            continue
        by_pair[row["pair"]].append((row["external_key"], row["value"]))
    return by_pair


def compute_gaps(pairs, choices, party):
    # chosen[(party_label, endorsement)] -> list of 0/1
    chosen = defaultdict(list)
    for pair_key, (a, b) in pairs.items():
        for ext_key, choice in choices.get(pair_key, []):
            p = party.get(ext_key)
            if p not in ("Republican", "Democrat"):
                continue
            for side_label, attrs in (("Vaccine A", a), ("Vaccine B", b)):
                chosen[(p, attrs["endorsement"])].append(1 if choice == side_label else 0)

    gaps = {}
    for endorsement in ENDORSEMENTS:
        rep = 100 * np.mean(chosen[("Republican", endorsement)])
        dem = 100 * np.mean(chosen[("Democrat", endorsement)])
        gaps[endorsement] = rep - dem
    return gaps


def main():
    pairs = load_pairs()
    party = load_party()
    # Values published in the paper's Table 4, for comparison.
    published = {
        "claude-haiku-4-5": {"Trump": 36.83, "Biden": -13.61, "CDC": 4.64, "WHO": -25.09},
        "gemini-3.7-flash": {"Trump": 32.26, "Biden": -27.31, "CDC": 7.17, "WHO": -25.90},
    }
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        choices = load_choices(model)
        gaps = compute_gaps(pairs, choices, party)
        print(f"\n{model}")
        for e in ENDORSEMENTS:
            pub = published[model][e]
            print(f"  {e:<8} recomputed {gaps[e]:+7.2f}   published {pub:+7.2f}   diff {gaps[e]-pub:+.2f}")


if __name__ == "__main__":
    main()
