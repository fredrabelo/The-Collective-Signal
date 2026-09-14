# The Collective Signal — Replication Package

Data, analysis code, and figures for the paper *"The Collective Signal:
Directional Preservation in Conjoint Experiments with Synthetic Populations"*
(submitted, 2026). The paper reconstructs the preregistered vaccine-choice
conjoint experiment of Kreps, Prasad, Brownstein, Hswen, Garibaldi, Zhang &
Kriner (2020), *"Factors Associated With US Adults' Likelihood of Accepting
COVID-19 Vaccination,"* *JAMA Network Open* 3(10):e2025594, and tests whether
synthetic LLM-based respondents preserve the direction, magnitude, ranking,
and decision-relevant conclusions of the human study under two complementary
instruments: an assignment-matched respondent replay and a shared 24-pair battery.

This repository does **not** include the underlying agent/prompting platform
used to generate the synthetic responses — see
[Reproducibility scope](#reproducibility-scope) below. Every synthetic-side
number and figure in the paper is independently reproducible from what's
released here, run the scripts below yourself to check. Every human-side
number is either directly reproducible (Table 2, from `human_choice` in
`matched_instrument_and_truth.json`), reproducible with the original Kreps
et al. Dataverse file that we do not redistribute (Tables 3-5), or given as
a derived aggregate statistic with a documented approximation that doesn't
require that file (Table 5's bootstrap). See
[Data integrity notes](#data-integrity-notes) below for specifics.

## Structure

```
data/            Instruments, personas, human ground truth, and all raw synthetic responses (JSON/CSV)
data/manifests/  Configuration for each model/condition (grounding compiler, covariates, execution timestamps)
analysis/        Scripts that turn data/ into every statistic and figure in the paper
figures/         Output of the analysis scripts — the exact PNGs used in the paper
export_replication_data.py   The SQL export that produced data/*_responses.csv
```

## Reproducing the paper's results

```bash
cd analysis
pip install numpy

# Figure 1: individual matched-choice fidelity (accuracy, balanced accuracy,
# per-class recall) in the respondent-matched replay.
python3 verify_matched_replay_fidelity.py

# Table 2 (Appendix Table 6): respondent-matched replay AMCEs, per contrast.
# The bootstrap here is JOINT: each resample draws respondents once and
# recomputes BOTH the human AMCE and the synthetic AMCE from that same
# resample, so the reported intervals incorporate human-side sampling
# uncertainty, not just synthetic-side uncertainty conditional on a fixed
# human benchmark (this is possible for Table 2, unlike Tables 3-5, because
# the matched-replay design pairs each synthetic respondent to a specific
# human respondent's own 5 pairs, so both sides are available per
# respondent). Also includes a paired bootstrap for the basic-vs-rich
# persona MAE difference within each model (RQ4), using the same joint
# resampling. Pass an integer to control the resample count (default 2000;
# the full run takes several minutes — roughly double
# compute_battery_amce.py's per-iteration cost, since it recomputes two
# AMCE vectors per resample instead of one).
python3 compute_matched_replay_amce.py [n_bootstrap]

# Table 3 (Appendix Table 7) / Figure 2: shared-battery AMCEs, parsed
# directly from the literal vignette text, with the same respondent-
# clustered bootstrap. The paper's estimator is OLS (all 7 attributes
# jointly); a simple difference-in-means is also reported as a sensitivity
# check. Pass an integer to enable the bootstrap (default 0 = point
# estimates only; the full run takes several minutes — this script is
# noticeably slower than compute_matched_replay_amce.py because each of the
# 24-pair battery's respondents contributes ~5x more rows per resample).
python3 compute_battery_amce.py [n_bootstrap]

# Table 4 / Figure 3: shared-battery Republican-minus-Democrat choice gaps
# by endorsement.
python3 compute_partisan_gaps.py

# Compares covariate distributions of Gemini's complete vs. incomplete
# shared-battery respondents, and checks whether restricting Claude's
# comparison to Gemini's completed subset changes anything.
python3 check_gemini_missingness.py

# Table 5: anchor-profile means, MAE, correlation, range, and the
# winner-selection bootstrap for both the synthetic side (fully
# respondent-level, from data/) and the human side. The human side uses a
# real respondent-level bootstrap if you provide the original Kreps et al.
# .dta (see the script's docstring for where to place it — not
# redistributed here); otherwise it falls back to a documented normal
# approximation from the published mean/sd/n, which closely matches the
# real bootstrap.
python3 compute_anchor_stats.py [n_bootstrap]

# Regenerates all four manuscript figures from the values these scripts
# produce (hardcoded here for figure styling; every value is independently
# reproducible by the script above that computes it, except the human side
# of Tables 3-4, which is a fixed external benchmark — see data/README.md).
python3 make_manuscript_figures.py   # -> ../figures/*.png, *.pdf
```

Each script is self-contained and reads only from `../data/`. No database,
API key, or external service is required to reproduce the paper's tables and
figures from the released data.

## Data integrity notes

A small number of malformed rows exist in the raw exports: one respondent's
Claude shared-battery reply contains two non-standard keys (`pair25_choice`,
`pair02_choice_explanation`) beyond the 24 real pairs, and one respondent's
Claude anchor-profile reply contains an extra key
(`anchor_99th_pct_likelihood_uk_origin`) beyond the five real anchors. Both
are excluded by `VALID_PAIR_KEYS`/equivalent filters in the scripts above and
do not affect any of the 24 real pairs or 5 real anchors for any respondent.
See `data/README.md` for exact counts.

## Reproducibility scope

The synthetic responses in `data/*_responses.csv` were generated using an
internal, not-publicly-released research platform, calling `claude-haiku-4-5`
and `gemini-3.7-flash` through their respective batch APIs. We do not release
that platform's source code. The paper's Appendix (Beyond PDF submission)
reproduces the complete, literal prompt text (persona rendering, vignette
template, and output schema) used to generate every response, which —
combined with the data in this repository — is sufficient to (a) verify every
synthetic-side statistic and figure in the paper (see the qualification on
human-side statistics above), and (b) regenerate equivalent synthetic
responses against the same models using only the Appendix's prompts and any
standard LLM API client.

## License

See `data/README.md` for the applicable licenses (they differ by file: the
Kreps et al. Dataverse release carries its own terms; newly generated data,
figures, and analysis code are CC BY 4.0 / MIT respectively).

## Citation

If you use this data or code, please cite both this paper (details to be
added on acceptance) and the original study:

```bibtex
@article{kreps2020factors,
  title={Factors Associated With {US} Adults' Likelihood of Accepting {COVID-19} Vaccination},
  author={Kreps, Sarah and Prasad, Sandip and Brownstein, John S. and Hswen, Yulin and Garibaldi, Brian T. and Zhang, Bao and Kriner, Douglas L.},
  journal={JAMA Network Open},
  volume={3}, number={10}, pages={e2025594}, year={2020},
  doi={10.1001/jamanetworkopen.2020.25594}
}
```
