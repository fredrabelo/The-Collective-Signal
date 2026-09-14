"""Recompute the respondent-matched replay's AMCEs (paper Table 2 / RQ1),
per contrast, with a joint respondent-clustered bootstrap — directly from
the raw responses and the literal instrument text.

Point estimates use the estimator the paper's Method section describes:
difference in mean choice between each level and its reference, with
"Neither" counted as zero for both profiles in a pair. Human-side AMCEs are
computed from matched_instrument_and_truth.json's `human_choice` field —
the real human choice on each respondent's own 5 realized pairs. They agree
with the 24-pair battery's human benchmark (Table 3) on 10 of 11 contrasts;
the WHO-endorsement contrast differs (5.75pp here vs. 7.75pp in Table 3)
because it is computed under a different (though overlapping) conditioning
of the same underlying human data.

Because the matched-replay design pairs each synthetic respondent to a
specific human respondent's own 5 pairs, human and synthetic choices are
both available for the same 1,971 respondents. Each bootstrap iteration
resamples respondents once, then recomputes both the human and synthetic
AMCE from that same resample before computing direction agreement, MAE, and
correlation between the two — this propagates human-side sampling
uncertainty into the fidelity statistics, rather than conditioning on the
human benchmark as fixed.

Usage: python3 compute_matched_replay_amce.py [n_bootstrap]
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from parse_vignette import ATTR_ORDER, REFERENCE, parse_pair

CONTRASTS = [
    ("efficacy", 70), ("efficacy", 90), ("duration", 5),
    ("major_side_effect", "1/1m"), ("minor_side_effect", "1/30"),
    ("fda_status", "emergency"), ("origin", "UK"), ("origin", "China"),
    ("endorsement", "Biden"), ("endorsement", "CDC"), ("endorsement", "WHO"),
]
CONDITIONS = [("claude-haiku-4-5", "basic_7cov"), ("claude-haiku-4-5", "rich_15cov"),
              ("gemini-3.7-flash", "basic_7cov"), ("gemini-3.7-flash", "rich_15cov")]


def load_instrument():
    """external_key -> list of 5 (attr_a, attr_b, human_choice) tuples."""
    data = json.load(open("../data/matched_instrument_and_truth.json"))
    out = {}
    for r in data:
        tasks = sorted(r["tasks"], key=lambda t: t["task_index"])
        out[r["external_key"]] = [
            (*parse_pair(t["question_text"]), t["human_choice"]) for t in tasks
        ]
    return out


def load_responses():
    out = defaultdict(lambda: defaultdict(dict))
    with open("../data/matched_replay_responses.csv") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["persona"])
            out[key][row["external_key"]][int(row["task_index"])] = row["value"]
    return out


def build_respondent_blocks(instrument, responses_for_condition):
    """Each respondent's block holds, per profile-side: attributes, whether
    the SYNTHETIC agent chose that side, and whether the HUMAN chose that
    side — both derived from the same 5 realized pairs, so a single
    resample of respondents carries both sides together."""
    blocks = {}
    for ext_key, tasks in instrument.items():
        choices = responses_for_condition.get(ext_key)
        if not choices or len(choices) != 5:
            continue
        rows = []
        for task_idx, (attr_a, attr_b, human_choice) in enumerate(tasks, start=1):
            synth_choice = choices.get(task_idx)
            if synth_choice is None:
                continue
            rows.append((attr_a, 1 if synth_choice == "Vaccine A" else 0, 1 if human_choice == "Vaccine A" else 0))
            rows.append((attr_b, 1 if synth_choice == "Vaccine B" else 0, 1 if human_choice == "Vaccine B" else 0))
        blocks[ext_key] = rows
    return blocks


def diff_in_means_amce(rows, outcome_index):
    """outcome_index: 1 for synthetic-chosen, 2 for human-chosen (matches
    the tuple layout in build_respondent_blocks)."""
    by_level = defaultdict(list)
    for r in rows:
        attrs, synth, human = r
        outcome = synth if outcome_index == 1 else human
        for attr in ATTR_ORDER:
            by_level[(attr, attrs[attr])].append(outcome)
    out = {}
    for attr in ATTR_ORDER:
        ref_mean = 100 * np.mean(by_level[(attr, REFERENCE[attr])])
        for c in CONTRASTS:
            if c[0] == attr:
                out[c] = 100 * np.mean(by_level[c]) - ref_mean
    return out


def compare(human_amce, synth_amce):
    human = np.array([human_amce[c] for c in CONTRASTS])
    synth = np.array([synth_amce[c] for c in CONTRASTS])
    directions = int(np.sum(np.sign(human) == np.sign(synth)))
    mae = float(np.mean(np.abs(human - synth)))
    corr = float(np.corrcoef(human, synth)[0, 1])
    return directions, mae, corr


def joint_bootstrap(blocks, n_boot, seed=42):
    """Each iteration resamples respondents once and recomputes BOTH human
    and synthetic AMCEs from that same resample (see module docstring)."""
    keys = list(blocks.keys())
    rng = np.random.default_rng(seed)
    n = len(keys)
    boot_human_amce = {c: [] for c in CONTRASTS}
    boot_synth_amce = {c: [] for c in CONTRASTS}
    boot_directions, boot_mae, boot_corr = [], [], []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        rows = []
        for i in sample_idx:
            rows.extend(blocks[keys[i]])
        human_amce = diff_in_means_amce(rows, outcome_index=2)
        synth_amce = diff_in_means_amce(rows, outcome_index=1)
        for c in CONTRASTS:
            boot_human_amce[c].append(human_amce[c])
            boot_synth_amce[c].append(synth_amce[c])
        d, m, r = compare(human_amce, synth_amce)
        boot_directions.append(d)
        boot_mae.append(m)
        boot_corr.append(r)
    return boot_human_amce, boot_synth_amce, boot_directions, boot_mae, boot_corr


def ci(values, lo=2.5, hi=97.5):
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def paired_richness_bootstrap(instrument, responses, model, n_boot, seed=42):
    """Paired bootstrap for the basic-vs-rich MAE difference within one
    model family. Each iteration resamples respondents once, recomputes the
    human AMCE from that SAME resample (shared between the basic and rich
    comparison, since both use the same respondents' human choices), and the
    synthetic AMCE separately for basic and rich — so the reported
    difference already accounts for human-side sampling uncertainty
    correlated across the two persona conditions within each iteration."""
    basic = build_respondent_blocks(instrument, responses[(model, "basic_7cov")])
    rich = build_respondent_blocks(instrument, responses[(model, "rich_15cov")])
    keys = sorted(set(basic) & set(rich))
    n = len(keys)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rows_b, rows_r = [], []
        for i in idx:
            k = keys[i]
            rows_b.extend(basic[k])
            rows_r.extend(rich[k])
        human_amce = diff_in_means_amce(rows_b, outcome_index=2)  # same resample -> same human rows either side
        mae_b = compare(human_amce, diff_in_means_amce(rows_b, outcome_index=1))[1]
        mae_r = compare(human_amce, diff_in_means_amce(rows_r, outcome_index=1))[1]
        diffs.append(mae_b - mae_r)
    all_rows_b = [r for k in keys for r in basic[k]]
    all_rows_r = [r for k in keys for r in rich[k]]
    point_human = diff_in_means_amce(all_rows_b, outcome_index=2)
    point_b = compare(point_human, diff_in_means_amce(all_rows_b, outcome_index=1))[1]
    point_r = compare(point_human, diff_in_means_amce(all_rows_r, outcome_index=1))[1]
    return point_b - point_r, ci(diffs)


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    instrument = load_instrument()
    responses = load_responses()

    print(f"{'Condition':<28}{'Directions':>16}{'MAE (pp)':>24}{'Correlation':>22}")
    condition_human_amce, condition_synth_amce = {}, {}
    condition_boot_human, condition_boot_synth = {}, {}
    for model, persona in CONDITIONS:
        blocks = build_respondent_blocks(instrument, responses[(model, persona)])
        all_rows = [row for rows in blocks.values() for row in rows]
        point_human = diff_in_means_amce(all_rows, outcome_index=2)
        point_synth = diff_in_means_amce(all_rows, outcome_index=1)
        directions, mae, corr = compare(point_human, point_synth)

        boot_human, boot_synth, boot_d, boot_mae, boot_corr = joint_bootstrap(blocks, n_boot)
        d_lo, d_hi = ci(boot_d)
        mae_lo, mae_hi = ci(boot_mae)
        corr_lo, corr_hi = ci(boot_corr)
        condition_human_amce[(model, persona)] = point_human
        condition_synth_amce[(model, persona)] = point_synth
        condition_boot_human[(model, persona)] = boot_human
        condition_boot_synth[(model, persona)] = boot_synth

        label = f"{model} {persona}"
        print(f"{label:<28}{directions:>3}/11 [{d_lo:.1f},{d_hi:.1f}]"
              f"{mae:>10.2f} [{mae_lo:.2f},{mae_hi:.2f}]"
              f"{corr:>10.3f} [{corr_lo:.3f},{corr_hi:.3f}]")

    print(f"\nPer-contrast point estimates and JOINT bootstrap 95% CIs (n_boot={n_boot}):\n")
    print("(Human column CI reflects resampling the same respondents used for that condition's synthetic column;")
    print(" it is not identical across conditions because different conditions have slightly different completion sets.)\n")
    for c in CONTRASTS:
        print(f"{c[0]:<20}{str(c[1]):<10}")
        for model, persona in CONDITIONS:
            h_point = condition_human_amce[(model, persona)][c]
            h_lo, h_hi = ci(condition_boot_human[(model, persona)][c])
            s_point = condition_synth_amce[(model, persona)][c]
            s_lo, s_hi = ci(condition_boot_synth[(model, persona)][c])
            print(f"    {model:<20}{persona:<12}human {h_point:>+7.2f} [{h_lo:>+6.2f},{h_hi:>+6.2f}]"
                  f"   synth {s_point:>+7.2f} [{s_lo:>+6.2f},{s_hi:>+6.2f}]")

    print(f"\nPaired bootstrap: basic-minus-rich MAE difference (n_boot={n_boot}):\n")
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        diff, (lo, hi) = paired_richness_bootstrap(instrument, responses, model, n_boot)
        excludes_zero = lo > 0 or hi < 0
        print(f"  {model:<20} basic-rich = {diff:+.2f}pp  95% CI [{lo:+.2f},{hi:+.2f}]"
              f"  {'(excludes zero)' if excludes_zero else '(includes zero)'}")


if __name__ == "__main__":
    main()
