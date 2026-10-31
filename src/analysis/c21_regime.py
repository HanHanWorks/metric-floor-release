# -*- coding: utf-8 -*-
"""C21 | Where inside the score's geometry does the gain sit?

The assessment rule's denominator switches from the metered output to a
constant at 0.2 of installed capacity. This script recomputes the score
separately on intervals below and above that threshold, for the 56 non-windowed
runs (c21_floor.csv) and for the 32 holdout runs (c21_ho.csv).

Reproduces Table 8 and Section 4.5.

Usage:
    python src/analysis/c21_regime.py [--json results/analysis/c21.json]
"""
from __future__ import print_function

import argparse
import json
import os

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TABLES = os.path.join(REPO, "data", "tables")


def _by_stratum(path):
    d = pd.read_csv(os.path.join(TABLES, path))
    per_run = d.groupby(["run", "stratum"]).dacc.mean().unstack()
    return d, per_run


def _pooled(d):
    """Mean score after minus mean score before, within each stratum.

    Identical to the mean of per-run means for every balanced subset here
    (each year, the holdout set); the two differ only when years with unequal
    run counts are pooled, where the paper reports this convention.
    """
    out = {}
    for stratum in ("low", "high"):
        x = d[d.stratum == stratum]
        out[stratum] = float(x.acc_new.mean() - x.acc_base.mean())
    return out


def compute():
    d, per_run = _by_stratum("c21_floor.csv")
    pooled = _pooled(d)

    out = dict(
        n_runs=int(len(per_run)),
        all_runs=dict(floor=pooled["low"], proportional=pooled["high"]),
        all_runs_run_mean=dict(floor=float(per_run.low.mean()),
                               proportional=float(per_run.high.mean())),
        by_year={},
        share_intervals_floor=float(d[d.stratum == "low"].n.sum() / d.n.sum()),
    )
    for year, g in d.groupby("year"):
        r = g.groupby(["run", "stratum"]).dacc.mean().unstack()
        out["by_year"][int(year)] = dict(
            n_runs=int(len(r)), floor=float(r.low.mean()),
            proportional=float(r.high.mean()),
            share_floor=float(g[g.stratum == "low"].n.sum() / g.n.sum()))

    rho, p = stats.spearmanr(per_run.low, per_run.high)
    out["within_run"] = dict(rho=float(rho), p=float(p))

    ho, per_run_ho = _by_stratum("c21_ho.csv")
    out["holdout"] = dict(n_runs=int(len(per_run_ho)),
                          floor=float(per_run_ho.low.mean()),
                          proportional=float(per_run_ho.high.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = compute()

    print("C21 | regime decomposition (dACC in points)")
    print("  floor regime holds %.1f%% of intervals\n" % (100 * res["share_intervals_floor"]))
    print("  %-18s %5s %10s %14s" % ("sample", "runs", "below floor", "above floor"))
    print("  %-18s %5d %+9.2f %13.2f" % ("all runs", res["n_runs"],
                                         res["all_runs"]["floor"],
                                         res["all_runs"]["proportional"]))
    for year in sorted(res["by_year"]):
        y = res["by_year"][year]
        print("  %-18d %5d %+9.2f %13.2f" % (year, y["n_runs"], y["floor"], y["proportional"]))
    ho = res["holdout"]
    print("  %-18s %5d %+9.2f %13.2f" % ("holdout runs", ho["n_runs"],
                                         ho["floor"], ho["proportional"]))
    w = res["within_run"]
    print("\n  within-run association between the two regime changes: "
          "rho=%.2f (p=%.2e)" % (w["rho"], w["p"]))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
