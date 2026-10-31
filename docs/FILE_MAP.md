# Where each number comes from

Every script, what it reads, what it produces, and where that lands in the
manuscript. Nothing in the paper's Results or Methods is outside this table.

## Tables

| table | script | reads | note |
|---|---|---|---|
| Table 1 (coverage) | — | — | computed from the restricted series; not in this repository |
| Table 2 (corpus) | `src/analysis/trace_tables.py` | `data/runs/meta_*.json` | `table4()` |
| Table 3 (four losses) | — | — | definitions only, no data |
| Table 4 (corpus and score change) | `src/analysis/trace_tables.py` | `data/runs/meta_*.json` | `table4()` |
| Table 5 (holdout designs) | `src/analysis/c20_holdout.py` | `data/tables/c20_holdout.csv` | exact Wilcoxon from `stats_exact.py` |
| Table 6 (by farm status) | `src/analysis/c19_station_ext.py` | `data/tables/c19_station_ext.csv` | |
| Table 7 (Diebold–Mariano) | `src/analysis/c22_dm.py` | `data/tables/c22_dm.csv` | unseen farms, n=960 per loss |
| Table 8 (regime decomposition) | `src/analysis/c21_regime.py` | `data/tables/c21_floor.csv`, `c21_ho.csv` | |
| Table 9 (mechanism prevalence) | `src/analysis/trace_tables.py` | `data/runs/code_*.py` + `src/loop/p3_ontology.py` | `table9()`; the extractor is the one that labelled the corpus |
| Table 10 (summary) | — | — | restates Tables 4–9 |
| Section 4.8 (floor-free counterfactual) | `server/nofloor_analysis.py` | `data/nf_runs/`, produced by `server/run_batch.sh` against the restricted series |

Table numbering follows the compiled manuscript. If the manuscript is
renumbered, this table is the thing to update.

## Figures

| figure | script | reads |
|---|---|---|
| Figure 1 (study design) | not generated | `figures/fig1_study_design.svg` — drawn by hand, not from data |
| Figure 2 (score and holdout) | `src/figures/make_fig2.py` | `data/runs/steps_*.csv`, `data/tables/c20_holdout.csv` |
| Figure 3 (cross-site) | `src/figures/make_fig3.py` | `data/tables/c19_station_ext.csv` |
| Figure 4 (two losses) | `src/figures/make_fig4.py` | `data/tables/c22_dm.csv` |
| Figure 5 (regime) | `src/figures/make_fig5.py` | `data/tables/c21_floor.csv`, `c21_ho.csv`, `c19_station_ext.csv`, `c24_lowshare.csv` |
| Figure 6 (what changed) | `src/figures/make_fig6.py` | `data/tables/c23_mech_prevalence.csv`, `c21_floor.csv`, `data/runs/steps_*.csv` |

Regenerating Figures 2–6 from this repository yields PDFs whose extracted text
and drawing primitives are identical to the ones in the submitted manuscript.

## Sections

| claim | where it is computed |
|---|---|
| +1.02 points, 106/106 runs above start, 22.4% acceptance | `trace_tables.table4()` |
| 79.3% of accepted edits add no new mechanism; first ten carry 72% | `trace_tables.marginal_returns()` |
| global rescaling +3.92 vs +1.08, floor operator 10.4% and p=0.60 | `trace_tables.mechanism_groups()` |
| 94.6% of runs gain on unseen farms; 64.3% worsen on squared error; ρ=0.589 | `c19_station_ext.compute()` |
| 90.5% / 38.8% of DM tests favour the pipeline under the assessment loss / squared error | `c22_dm.compute()` |
| +3.11 below the floor, −0.91 above; +5.38/−2.10 in the holdout runs; ρ=−0.54 | `c21_regime.compute()` |
| exact Wilcoxon p=0.0092 under the farm split | `c20_holdout.compute()` |

## Files not carried over

Two exploratory threads from the analysis log are deliberately excluded, because
the manuscript does not report them: the selection-null simulation (`C16c`) and
the novel-combination gain check. Their tables remain in the project archive,
not here.
