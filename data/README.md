# Data released with this repository

Three kinds of data are here: the **artefacts of each floored run** (`runs/`),
the **floor-free counterfactual** (`nf_runs/`), and the **result tables** built
from them (`tables/`). None contains metered output or issued forecasts — only
scores, errors and the code the loop wrote.

## `runs/` — 106 runs, 4 files each

| file | rows | contents |
|---|---|---|
| `meta_<run>.json` | 1 | `run`, `year`, `memory`, `seed`, `n_iter`, `acc_first`, `acc_last`, `n_accept`, `n_pure_rediscovery`, `wall_sec` |
| `steps_<run>.csv` | 1 per candidate that produced a score | `run, it, accepted, acc, mechs, n_overlap, n_novel, pure_rediscovery` |
| `trace_<run>.csv` | 1 per iteration | `run, year, memory, seed, it, mase, acc, pen, n_mech, mechs`: the state the loop was in after that iteration (score, scale-free error, penalised energy, mechanism count and set) |
| `code_<run>.py` | 1 | the final accepted `transform(view)` for that run |

`steps` holds 5,200 rows across the corpus rather than 5,360: candidates whose
code failed to execute produced no score and were not written. They are still
counted in `n_iter`.

Run identifiers: `Y<year>_<arm>_s<seed>`. Arms are `nomem`, `mem`, `full`,
`none`, `regime` (knowledge-injection policies) and `win_<yymmdd>` (windowed
campaigns, excluded from every analysis in the paper).

## `nf_runs/` — the floor-free counterfactual

128 artefacts from the 32 counterfactual runs of Section 4.8 (four arms x eight
seeds, 2025, 60 iterations, denominator = installed capacity): the same four file
kinds as `runs/`, with tags `NFY2025_<arm>_base_s<seed>`, plus the two summary
files `nf_summary.json` and `nf_rescored.csv` (each floor-free pipeline re-scored
under the floored rule). Mean change under the floor-free objective: +0.05 points
against +1.09 for the floored corpus on the same arms, seeds and farms.

## `tables/` — result tables

All seven were produced on the compute node that held the restricted series.
They are released here as inputs to the analysis scripts, not as data that can
be rebuilt from this repository.

### `c19_station_ext.csv` — 1,184 rows (56 runs x 14–24 farms)

Cross-site transfer. Every final pipeline re-applied, unmodified, to every
eligible farm in its year of origin.

| column | meaning |
|---|---|
| `run`, `year`, `seed` | which pipeline |
| `station` | pseudonymous grid identifier for the wind farm |
| `seen` | 1 if the farm was among the four used in the search, 0 otherwise |
| `cap` | installed capacity, MW |
| `n_days` | days with data for that farm |
| `acc_base`, `acc_new` | assessment score before / after, points |
| `dacc` | `acc_new - acc_base` |
| `mase_base`, `mase_new` | scale-free error (mean absolute error divided by the mean absolute day-to-day change), before / after |
| `pen_base`, `pen_new` | penalised energy before / after |
| `err` | reserved for reconciliation notes; empty throughout this release |

### `c20_holdout.csv` — 32 rows

The honest-selection experiment: acceptance decided on one subset, reported on
another.

| column | meaning |
|---|---|
| `run`, `mode`, `seed` | `mode` is `station` (farms split 2/2) or `time` (days split 60/40) |
| `n_accept`, `tol` | edits accepted, acceptance margin `tau` |
| `acc_val_first`, `acc_val_last` | score on the validation subset, first and last iteration |
| `acc_test_first`, `acc_test_last` | score on the held-out subset, first and last iteration |
| `d_val`, `d_test` | the corresponding changes |

The 32 holdout runs are not in `runs/`: their per-iteration traces were produced
in a separate campaign and only the summary above was retained.

### `c21_floor.csv` (2,360 rows) and `c21_ho.csv` (1,536 rows)

Regime decomposition: the assessment score recomputed separately on intervals
below and above `0.2 x cap`.

| column | meaning |
|---|---|
| `run`, `year`, `station` | the run x farm pair |
| `stratum` | `low` (metered output below the floor) or `high` |
| `n` | intervals in that stratum |
| `acc_base`, `acc_new`, `dacc` | score before, after, change, within the stratum |

`c21_floor.csv` holds the 56 non-windowed runs; `c21_ho.csv` the 32 holdout runs.

### `c22_dm.csv` — 3,552 rows (1,184 pairs x 3 losses)

Diebold–Mariano tests on daily loss differentials.

| column | meaning |
|---|---|
| `run`, `year`, `seed`, `station`, `seen` | the run x farm pair |
| `loss` | `metric` (the assessment rule's own loss), `se` (capacity-normalised squared error), `ae` (capacity-normalised absolute error) |
| `n_days` | daily differentials entering the test |
| `mean_d`, `p`, `rel` | mean loss differential, two-sided p-value, relative gain |
| `dm` | DM statistic, Newey–West long-run variance with bandwidth `floor(4 (N/100)^(2/9))`, Harvey–Leybourne–Newbold correction. Positive = the pipeline is better |

### `c23_mech_prevalence.csv` — 106 rows

Mechanism classes present in each final pipeline, as extracted from
`runs/code_<run>.py` by `src/loop/p3_ontology.py`. Rebuildable from this
repository: `trace_tables.py` does exactly that and prints Table 9.

### `c24_lowshare.csv` — 57 rows (farm x year)

| column | meaning |
|---|---|
| `year`, `station`, `cap`, `n` | farm-year, capacity, intervals recorded |
| `low_share` | share of intervals below `0.2 x cap` |
| `zero_share` | share of intervals at zero output |

## What is withheld, and why

The metered output series and the issued forecasts. They are the property of the
dispatching utility, are governed by a data-sharing agreement that does not
permit redistribution, and are **confidential**; they cannot be released, and no
de-identified sample is included. Nothing in the released tables above can be
inverted to recover them, since only scores and aggregated errors survive.

Farm identifiers are the grid's own numeric codes. No names, coordinates or
owner information are included.
