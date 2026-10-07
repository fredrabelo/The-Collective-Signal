"""Cross-estimator sensitivity check for the two conjoint instruments.

For each model/persona condition in the assignment-matched replay, report
human-versus-synthetic fidelity using both (a) simple differences in means
and (b) OLS adjusted simultaneously for all seven randomized attributes.
For the shared battery, report the same two estimators using the published
human benchmark.  This separates estimator sensitivity from the observed
difference between instruments; it does not identify a causal instrument
effect because the human benchmark and pair conditioning also differ.

Usage: python3 check_estimator_sensitivity.py
"""
from compute_battery_amce import (
    HUMAN_AMCE,
    build_long_table,
    load_choices,
    load_pairs,
    ols_adjusted_amce,
    simple_diff_amce,
    stats_from_amce,
)
from compute_matched_replay_amce import (
    CONDITIONS,
    build_respondent_blocks,
    compare,
    diff_in_means_amce,
    load_instrument,
    load_responses,
)


def tuple_rows_to_dicts(rows, outcome_index):
    """Convert matched-replay tuples to the long-table format used by OLS."""
    return [dict(attrs, chosen=row[outcome_index]) for row in rows for attrs in [row[0]]]


def print_replay():
    instrument = load_instrument()
    responses = load_responses()
    print("ASSIGNMENT-MATCHED REPLAY")
    print(f"{'Condition':<42}{'Estimator':<14}{'Directions':>11}{'MAE':>10}{'r':>9}")
    for model, persona in CONDITIONS:
        blocks = build_respondent_blocks(instrument, responses[(model, persona)])
        rows = [row for block in blocks.values() for row in block]

        human_dim = diff_in_means_amce(rows, outcome_index=2)
        synth_dim = diff_in_means_amce(rows, outcome_index=1)
        d, mae, corr = compare(human_dim, synth_dim)
        label = f"{model} / {persona}"
        print(f"{label:<42}{'Diff. means':<14}{d:>8}/11{mae:>10.2f}{corr:>9.3f}")

        human_ols = ols_adjusted_amce(tuple_rows_to_dicts(rows, outcome_index=2))
        synth_ols = ols_adjusted_amce(tuple_rows_to_dicts(rows, outcome_index=1))
        d, mae, corr = compare(human_ols, synth_ols)
        print(f"{'':<42}{'OLS':<14}{d:>8}/11{mae:>10.2f}{corr:>9.3f}")


def print_battery():
    pairs = load_pairs()
    print("\nSHARED 24-PAIR BATTERY")
    print(f"{'Condition':<42}{'Estimator':<14}{'Directions':>11}{'MAE':>10}{'r':>9}")
    for model in ("claude-haiku-4-5", "gemini-3.7-flash"):
        rows = build_long_table(pairs, load_choices(model))
        for label, amce in (
            ("Diff. means", simple_diff_amce(rows)),
            ("OLS", ols_adjusted_amce(rows)),
        ):
            d, mae, corr = stats_from_amce(amce)
            model_label = model if label == "Diff. means" else ""
            print(f"{model_label:<42}{label:<14}{d:>8}/11{mae:>10.2f}{corr:>9.3f}")
    print("\nBattery statistics compare with the published human benchmark:")
    print(", ".join(f"{key[0]}={value:+.2f}" for key, value in HUMAN_AMCE.items()))


if __name__ == "__main__":
    print_replay()
    print_battery()
