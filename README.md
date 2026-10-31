# Optimising the metric, not the forecast

Code and data release for the manuscript *"Optimising the metric, not the
forecast: how an automated loop exploits a grid-code accuracy score"*.

Three years of an automated post-processing loop on a provincial Chinese wind
fleet — 106 runs, 5,360 iterations, 1,202 accepted code edits. The loop raised
the assessment score by 1.02 points and the gain survived a change of subset, a
change of site and both holdout designs. Under a loss that does not share the
score's denominator it did not survive: the score rose while capacity-normalised
squared error worsened in 64% of runs.

This repository contains the final pipelines the loop produced, the per-iteration
traces, the farm-level result tables, and every script that turns those tables
into the numbers, tables and figures in the paper.

## What reproduces, exactly

| | |
|---|---|
| `python replication/verify_paper_numbers.py` | re-checks **134 printed numbers** (Tables 4–9, Sections 4.1–4.6) against the released tables: 134/134 pass |
| `make figures` | regenerates Figures 2–6; the PDFs are **content-identical** to the ones in the manuscript |
| `python src/analysis/trace_tables.py` | rebuilds Table 4 and Table 9 from the raw run artefacts (metadata, per-candidate steps, final code) |
| `python src/analysis/c19_station_ext.py` … `c22_dm.py` | rebuild the four farm-level tables |

Three printed values differ from what the released tables give, all by 0.01.
They are listed in [`docs/KNOWN_DISCREPANCIES.md`](docs/KNOWN_DISCREPANCIES.md)
and flagged as `diff` in the verification output rather than silently tolerated.

## What does not reproduce

The **search itself**. Re-running the loop needs the metered output and the
issued day-to-day forecasts for 14–24 wind farms over three years. Those series
are commercially restricted and are not distributed; a de-identified sample and
an application procedure are described in the manuscript's data-availability
statement. `src/loop/` therefore documents exactly how the loop was run — the
prompt, the sandbox the candidate code executes in, the acceptance rule, the
twelve-class mechanism extractor — but cannot be executed end to end without
that data.

Everything downstream of the loop is reproducible here, which is the part that
carries the paper's claims: given the same forecasts and the same final
pipelines, the regime decomposition, the Diebold–Mariano tests and the
cross-site transfer are arithmetic.

## Quick start

```bash
git clone <this repository> && cd metric-floor-release
python -m pip install -r requirements.txt

python replication/verify_paper_numbers.py     # 134 checks, ~10 s
python src/analysis/trace_tables.py            # Tables 4 and 9 on stdout
make figures                                   # Figures 2-6 into figures/
```

Tested with Python 3.7 and pandas 1.1.5 / numpy 1.21 / scipy 1.1 / matplotlib
2.2. Nothing newer than pandas 1.1 is required; the exact tests are implemented
in `src/analysis/stats_exact.py` so results do not depend on the scipy build.

## Layout

```
data/runs/          106 runs x 4 artefacts  (code, meta, steps, trace)
data/tables/        farm-level and run-level result tables (7 csv)
src/loop/           the search: agent, ontology, metric core (needs withheld data)
src/analysis/       tables 4-9 from the released artefacts
src/figures/        figures 2-6
replication/        the 129-number check
figures/            generated figures, plus the Figure 1 design as SVG
docs/               file map, known discrepancies
```

`data/README.md` documents every column. `docs/FILE_MAP.md` maps each file to
what it produces and where that appears in the paper.

## The runs in one line

`Y2023_nomem_s0` reads as: year 2023, no mechanism memory injected into the
prompt, seed 0. `win_<yymmdd>` marks a windowed campaign run (reserved for a
separate stability analysis and excluded from every analysis here). 56 of the
106 runs are non-windowed.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use these data or this code, please
cite the manuscript and this archive (DOI minted by Zenodo on release).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data tables and run artefacts: CC BY 4.0
([`LICENSE-DATA`](LICENSE-DATA)).
