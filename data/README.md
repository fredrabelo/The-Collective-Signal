# Data

## Files

**Instruments and ground truth (inputs to the LLMs, plus human ground truth):**

- **`respondents_basic.json`** / **`respondents_rich.json`** — one entry per
  respondent (N=1,971), each with an `external_key` and an `evidence` list of
  demographic facts. `respondents_basic.json` has the 7-covariate persona
  (age, gender, race, party, ideology, education, income); `respondents_rich.json`
  adds 8 further covariates (state, religion, evangelical identity, insurance,
  flu-vaccination frequency, pharmaceutical-industry favorability, personal
  contact with COVID-19, view of pandemic trajectory). No respondent outcome
  is ever included. Values are real, de-identified attributes recoded from the
  public Kreps et al. (2020) replication file — nothing is invented or
  inferred.
- **`matched_instrument_and_truth.json`** — the respondent-matched five-pair
  replay instrument: for each of the 1,971 respondents, the exact five
  vaccine-profile pairs their matched human saw (literal vignette text) and
  that human's realized choice (`human_choice`) for each pair. This is the
  primary experiment's instrument and ground truth (paper Table 2 / RQ1).
- **`shared_battery_instrument.json`** — the 24 canonical pairs used in the
  complementary shared-instrument analysis (paper Table 3 / RQ2), sampled
  systematically from pair combinations that occurred in the real human data.
- **`anchor_profile_instrument.json`** / **`anchor_profile_human_benchmark.json`**
  — the five source-defined anchor profiles (1st/25th/50th/75th/99th
  percentile of predicted human acceptance) and the real observed human
  ratings for each: `mean`, `n` (N=24–37 per profile), and `sd` — a derived
  aggregate statistic enabling a normal-approximation bootstrap of the human
  side without redistributing raw ratings, see `analysis/compute_anchor_stats.py`
  (paper Table 5 / RQ3).
- **`manifests/`** — the configuration for each model/condition: model,
  provider, grounding compiler, covariate list, and execution timestamps.

**Synthetic responses (LLM outputs, one row per choice/rating):**

- **`matched_replay_responses.csv`** — `model, persona, external_key,
  task_index, value, explanation`. 9,855 rows per condition (4 conditions:
  Claude/Gemini × basic/rich persona), one row per respondent × pair.
- **`shared_battery_responses.csv`** — `model, external_key, pair, value,
  explanation`. One row per respondent × canonical pair. Gemini: 45,288 rows
  (1,887 × 24), a shortfall from truncated-JSON batch failures on 84/1,971
  respondents, documented in the paper's Method section. Claude: 47,306 rows
  (1,971 × 24, plus 2 non-standard rows — `pair25_choice` and
  `pair02_choice_explanation` — from a single respondent's malformed batch
  reply; both fall outside the 24 real pair keys and are excluded by every
  script in `../analysis/`).
- **`anchor_profile_responses.csv`** — `model, external_key, anchor_item,
  value, explanation`. One row per respondent × anchor profile. Gemini: 9,780
  rows (1,956 × 5), same truncation cause as above. Claude: 9,856 rows
  (1,971 × 5, plus 1 non-standard row — `anchor_99th_pct_likelihood_uk_origin`
  — from a single respondent's malformed reply; excluded the same way).

All three response files were exported directly from the two internal
research-platform databases with `../export_replication_data.py` — a
straight SQL export with no proprietary grounding-compiler logic — so the
CSVs are exactly what the paper's statistics were computed from.

**What is deliberately not included:** an initial cell-level comparison of
the 24-pair battery against a *model-based* human benchmark (constructed by
fitting an additive model to the real data, not by direct observation) was
judged unsuitable as a validation target during study development and is not
used to support any conclusion in the paper (see the paper's Method section,
footnote on the shared battery). It is omitted here to avoid it being mistaken
for a validated ground truth.

## Source

Human ground truth: Kreps, Prasad, Brownstein, Hswen, Garibaldi, Zhang &
Kriner (2020), "Factors Associated With US Adults' Likelihood of Accepting
COVID-19 Vaccination," *JAMA Network Open* 3(10):e2025594, DOI
`10.1001/jamanetworkopen.2020.25594`. Replication data: Harvard Dataverse, DOI
`10.7910/DVN/6BSJYP`. Please cite both the paper and the original Dataverse
release as a matter of scientific attribution.

This repository does not redistribute the original Kreps et al. Stata
microdata (`.dta`) itself. The human side of Table 2 (matched replay) is
fully reproducible from what is released here (`human_choice` per task in
`matched_instrument_and_truth.json`). The human side of Table 3 (shared
battery) and Table 4 (partisan gaps) is a fixed benchmark computed from the
original 19,710-row human dataset and is not resampled here; recomputing it,
or its own sampling uncertainty, requires the original microdata. The human
side of Table 5 (anchors) can be approximated without the original file via
`anchor_profile_human_benchmark.json`'s published mean/sd/n
(`analysis/compute_anchor_stats.py` documents both modes); place the `.dta`
at the path that script expects for an exact respondent-level bootstrap
instead. Obtain the original microdata directly from Harvard Dataverse if
you need any of this.

## Licensing and terms of use

- The Kreps et al. replication package (Harvard Dataverse, DOI
  `10.7910/DVN/6BSJYP`) is subject to its own stated terms; consult the
  Dataverse listing before redistribution.
- The newly generated synthetic responses, derived respondent attributes, and
  original figures/tables in this repository are released under the Creative
  Commons Attribution 4.0 International (CC BY 4.0) License.
- Analysis and export code (`export_replication_data.py`,
  `analysis/*.py`) is released under the Massachusetts Institute of
  Technology (MIT) License.
