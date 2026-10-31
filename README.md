# Optimising the metric, not the forecast

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23837560.svg)](https://doi.org/10.5281/zenodo.23837560)

Code and data release for the manuscript *"Optimising the metric, not the
forecast: how an automated loop exploits a grid-code accuracy score"*.

Three years of an automated post-processing loop on a provincial Chinese wind
fleet — 106 runs, 5,360 iterations, 1,202 accepted code edits. The loop raised
the assessment score by 1.02 points and the gain survived a change of subset, a
change of site and both holdout designs. Under a loss that does not share the
score's denominator it did not survive: the score rose while capacity-normalised
squared error worsened in 64% of runs.

This repository contains the final pipelines the loop produced, the per-iteration
traces, the farm-level result tables, the 32-run floor-free counterfactual
reported in Section 4.8, and every script that turns those tables into the
numbers, tables and figures in the paper.

## Data confidentiality

The metered output and issued forecast series are confidential: they belong to
the dispatching utility, are governed by a data-sharing agreement that does not
permit redistribution, and are not included in this archive. No de-identified
sample is available, and requests for access cannot be granted. The released
artefacts are the final pipelines, per-iteration traces, aggregate result
tables and analysis code; none can be inverted to recover the raw series.

## What reproduces, exactly

| | |
|---|---|
| `python replication/verify_paper_numbers.py` | re-checks **142 printed numbers** (Tables 4–9, Sections 4.1–4.8) against the released tables: 142/142 pass |
| `make figures` | regenerates Figures 2–6; the PDFs are **content-identical** to the ones in the manuscript |
| `python src/analysis/trace_tables.py` | rebuilds Table 4 and Table 9 from the raw run artefacts (metadata, per-candidate steps, final code) |
| `python src/analysis/c19_station_ext.py` … `c22_dm.py` | rebuild the four farm-level tables |
| `python src/analysis/revision_checks.py` | the checks added in revision: farm reuse, two-way clustered inference, counting flow |

The first submission printed three values that the tables do not support, all
by 0.01. They are corrected in the current manuscript and recorded in
[`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md).

Three analyses need the withheld series to rebuild from raw data: the exact
region decomposition, the reconciliation audit, and the single-parameter blind
search. Their reported values are in the manuscript; the protocols that
reproduce them on a machine with the restricted data ship in
[`server/`](server/README.md).

## What does not reproduce

The **search itself**. Re-running the loop needs the metered output and the
issued day-to-day forecasts for 14–24 wind farms over three years. Those series
are **confidential**: owned by the provincial system operator and governed by a
data-sharing agreement that does not permit redistribution. They are not
distributed, and no farm-level series can be reconstructed from this archive.
No de-identified sample is included, and requests for access cannot be granted. `src/loop/` therefore documents exactly how the loop was run — the
prompt, the sandbox the candidate code executes in, the acceptance rule, the
twelve-class mechanism extractor — but cannot be executed end to end without
that data.

Everything downstream of the loop is reproducible here, which is the part that
carries the paper's claims: given the same forecasts and the same final
pipelines, the regime decomposition, the Diebold–Mariano tests and the
cross-site transfer are arithmetic.

## Quick start

```bash
git clone https://github.com/HanHanWorks/metric-floor-release.git && cd metric-floor-release
python -m pip install -r requirements.txt

python replication/verify_paper_numbers.py     # 142 checks, ~10 s
python src/analysis/trace_tables.py            # Tables 4 and 9 on stdout
make figures                                   # Figures 2-6 into figures/
```

Tested with Python 3.7 and pandas 1.1.5 / numpy 1.21 / scipy 1.1 / matplotlib
2.2. Nothing newer than pandas 1.1 is required; the exact tests are implemented
in `src/analysis/stats_exact.py` so results do not depend on the scipy build.

## Layout

```
data/runs/          106 floored runs x 4 artefacts  (code, meta, steps, trace)
data/nf_runs/       32 floor-free runs x 4 artefacts + summary files
data/tables/        farm-level and run-level result tables (7 csv)
src/loop/           the search: agent, ontology, metric core (needs withheld data)
src/analysis/       tables 4-9 from the released artefacts
src/figures/        figures 2-6
replication/        the 142-number check
figures/            generated figures, plus the Figure 1 design as SVG
docs/               file map, known discrepancies
```

`data/README.md` documents every column. `docs/FILE_MAP.md` maps each file to
what it produces and where that appears in the paper.

## The runs in one line

`Y2023_nomem_s0` reads as: year 2023, no mechanism memory injected into the
prompt, seed 0. `win_<yymmdd>` marks a windowed campaign run (reserved for a
separate stability analysis and excluded from every analysis here). 56 of the
106 floored runs are non-windowed. `NFY2025_<arm>_base_s<seed>` marks the 32
floor-free counterfactual runs of Section 4.8.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use these data or this code, please
cite the manuscript and this archive.

- Version DOI (v1.0.3): [10.5281/zenodo.23837560](https://doi.org/10.5281/zenodo.23837560)
- Concept DOI (all versions): [10.5281/zenodo.23837559](https://doi.org/10.5281/zenodo.23837559)

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data tables and run artefacts: CC BY 4.0
([`LICENSE-DATA`](LICENSE-DATA)).
