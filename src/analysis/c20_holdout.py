# -*- coding: utf-8 -*-
"""C17 | Honest selection: does a gain selected on one subset survive another?

Reads data/tables/c20_holdout.csv: 32 runs, 16 with farms split 2/2 and 16 with
days split 60/40. Acceptance used the validation subset only; the reported
change is on the held-out subset.

Reproduces Table 5 and Section 4.2.

Usage:
    python src/analysis/c20_holdout.py [--json results/analysis/c20.json]
"""
from __future__ import print_function

import argparse
import json
import os

import pandas as pd
from scipy import stats

from stats_exact import wilcoxon_exact

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TABLES = os.path.join(REPO, "data", "tables")


def compute():
    h = pd.read_csv(os.path.join(TABLES, "c20_holdout.csv"))
    out = {}
    for mode, g in h.groupby("mode"):
        t = stats.ttest_1samp(g.d_test, 0)
        paired = stats.ttest_rel(g.d_val, g.d_test)
        w = wilcoxon_exact(g.d_test)
        out[mode] = dict(
            n=int(len(g)),
            d_accept_mean=float(g.d_val.mean()), d_accept_sd=float(g.d_val.std(ddof=1)),
            d_holdout_mean=float(g.d_test.mean()), d_holdout_sd=float(g.d_test.std(ddof=1)),
            ratio=float(g.d_test.mean() / g.d_val.mean()),
            t=float(t.statistic), p=float(t.pvalue),
            wilcoxon_p=float(w["p"]), wilcoxon_method=w["method"],
            paired_t=float(paired.statistic), paired_p=float(paired.pvalue),
            n_accept_mean=float(g.n_accept.mean()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = compute()

    print("C17 | holdout designs (positive = final pipeline above its start)")
    print("  %-8s %3s %16s %16s %6s %18s %20s" % (
        "split", "n", "accept subset", "held-out", "ratio", "vs 0", "paired vs accept"))
    for mode in ("station", "time"):
        r = res[mode]
        print("  %-8s %3d  %+.2f (%.2f)    %+.2f (%.2f)   %.2f  t=%.2f p=%.4f  t=%.2f p=%.4f" % (
            mode, r["n"],
            r["d_accept_mean"], r["d_accept_sd"],
            r["d_holdout_mean"], r["d_holdout_sd"], r["ratio"],
            r["t"], r["p"], r["paired_t"], r["paired_p"]))
        print("           exact Wilcoxon of the held-out change: p=%.4g (%s)" % (
            r["wilcoxon_p"], r["wilcoxon_method"]))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
