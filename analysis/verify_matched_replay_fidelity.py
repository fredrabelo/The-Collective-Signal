"""Independently recompute the respondent-matched replay's individual
matched-choice fidelity (accuracy, balanced accuracy, per-class recall) from
the raw exported responses and ground truth, to verify the paper's Figure 1
/ RQ1 numbers directly from released data (no dependency on the internal
platform's own aggregation code).

Usage: python3 verify_matched_replay_fidelity.py
Reads: ../data/matched_replay_responses.csv, ../data/matched_instrument_and_truth.json
"""
import csv
import json
from collections import defaultdict

CATS = ["Vaccine A", "Vaccine B", "Neither"]


def load_truth(path):
    truth = json.load(open(path))
    lookup = {}
    for r in truth:
        for t in r["tasks"]:
            lookup[(r["external_key"], t["task_index"])] = t["human_choice"]
    return lookup


def main():
    truth = load_truth("../data/matched_instrument_and_truth.json")

    by_condition = defaultdict(list)  # (model, persona) -> [(pred, true), ...]
    with open("../data/matched_replay_responses.csv") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["persona"])
            human = truth.get((row["external_key"], int(row["task_index"])))
            if human is None:
                continue
            by_condition[key].append((row["value"], human))

    print(f"{'Condition':<28}{'N':>8}{'Raw acc.':>10}{'Bal. acc.':>11}"
          f"{'Recall A':>10}{'Recall B':>10}{'Recall Neither':>16}")
    for (model, persona), pairs in sorted(by_condition.items()):
        n = len(pairs)
        correct = sum(1 for p, h in pairs if p == h)
        raw_acc = 100 * correct / n

        recall = {}
        for cat in CATS:
            true_n = sum(1 for p, h in pairs if h == cat)
            true_correct = sum(1 for p, h in pairs if h == cat and p == cat)
            recall[cat] = 100 * true_correct / true_n if true_n else float("nan")
        bal_acc = sum(recall.values()) / len(CATS)

        label = f"{model} {persona}"
        print(f"{label:<28}{n:>8}{raw_acc:>10.2f}{bal_acc:>11.2f}"
              f"{recall['Vaccine A']:>10.2f}{recall['Vaccine B']:>10.2f}"
              f"{recall['Neither']:>16.2f}")


if __name__ == "__main__":
    main()
