# Scripts that need the restricted series

The analyses below need the metered and issued-forecast series, which are
confidential and not distributed. They are here as runnable protocols so that a
reader with access can execute exactly what the paper promises. Their reported
values are in Section 2.3, Equation (3), Table 8, Section 4.7 and Section 4.8
of the manuscript.

None of the five is covered by CI, because each needs the archive. They were run on
the compute node for the revision; the numbers they produced are the ones reported
in the manuscript.
They compile and import against the released loop code, but their only input is
the archive. Set `WIND_DATA_ROOT` before using them.

| script | answers | output |
|---|---|---|
| `decompose_floor.py` | How much of the change in the score is attributable to each region? | `decomp_<year>.csv`, per run × farm: total change, the two contributions, and a check that they sum identically |
| `reconcile_audit.py` | Which combination of forecast source, power convention and capacity vintage reproduces the operator's annual table? | `recon_<year>.csv`, one row per farm × variant, with the distance from the official value |
| `blind_search.py` | Can a structure-blind optimiser---one multiplicative constant, chosen by grid or at random---recover the loop's gain? | `blind_<year>.csv`, per run × farm: the best constant, the gain from it, the gain from a random draw, and the gain from one constant shared by the run |
| `reconcile_archive.py` | How far is our recomputation from the operator's annual table, fleet-wide? | `recon_archive.csv` + `recon_summary.json`: per farm and year, median absolute difference, share within 1 and 3 points, correlation |
| `caliber_sources.py` | Does the second forecast series in the archive reconcile better? | `caliber_sources_<year>.csv/.json`: MAE, bias, correlation and coverage per `data_source` |

## Why the decomposition is not in the paper

Eq. (2) of the manuscript defines region sub-scores and states that they do not
partition the total: the rule averages a within-day root mean square, so a
region enters the total nonlinearly. `decompose_floor.py` computes the
decomposition that *does* partition it, using

    dACC = mean_d(D0_d - D1_d),   D_d = sqrt(w_Fd A_Fd + w_Pd A_Pd)

and, because `x - y = (x^2 - y^2)/(x + y)`,

    c_R = mean_d[ w_Rd (A_R0d - A_R1d) / (D0_d + D1_d) ],   c_F + c_P = dACC

identically. The script prints `max |identity error|`, which should be
floating-point noise; if it is not, the weights differ between the two versions
and the comparison is invalid.

## Why the reconciliation is a bound, not an identification

The manuscript reports four farms: two agree with the operator's table to within
a point, two differ by 21-23 points in opposite directions. It argues that an
available-power convention, applied in the denominator only, can only *raise* the
recomputed score, so it can explain at most one of the two. `reconcile_audit.py`
turns that argument into a test: it crosses every forecast source in the archive
with three power conventions and two capacity vintages, and names the cell that
reproduces the official number for each farm.

The script expects two optional files whose column names must be checked against
the archive: available power (`farm_id, data_time, available_power`) and monthly
capacity (`farm_id, month, capability`). Without them the corresponding variants
are skipped and the output says so.

## Floor-free counterfactual

The counterfactual is reported in Section 4.8 of the manuscript.
`make_floorfree.sh` rebuilds the environment it used (three edits, listed in the
script), `run_batch.sh` runs the 32 counterfactual runs, and
`nofloor_analysis.py` produces the comparison with the floored corpus. The
counterfactual's 128 artefacts ship in `data/nf_runs/`.
`src/loop/` contains everything needed to run it: point `--accept` and the
scoring function at a floor-free rule and the same apparatus produces the
counterfactual trace.
