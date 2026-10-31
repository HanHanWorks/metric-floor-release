# -*- coding: utf-8 -*-
"""C19 | Transfer to farms that took no part in the search.

Reads data/tables/c19_station_ext.csv: one row per run x farm, holding the
assessment score and the scale-free error (MASE) before and after the loop,
on the same forecasts.

Reproduces Table 6 and Section 4.3.

Usage:
    python src/analysis/c19_station_ext.py [--json results/analysis/c19.json]
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


def compute():
    d = pd.read_csv(os.path.join(TABLES, "c19_station_ext.csv"))
    ok = d[d.dacc.notna()].copy()
    ok["dmase_pct"] = 100.0 * (ok.mase_new - ok.mase_base) / ok.mase_base

    out = dict(n_rows=int(len(ok)), n_runs=int(ok.run.nunique()), by_year={})
    for year, g in ok.groupby("year"):
        s, u = g[g.seen == 1], g[g.seen == 0]
        out["by_year"][int(year)] = dict(
            n_farms=int(g.station.nunique()), n_runs=int(g.run.nunique()),
            seen=dict(n=int(len(s)), dacc=float(s.dacc.mean()),
                      dmase_pct=float(s.dmase_pct.mean())),
            unseen=dict(n=int(len(u)), dacc=float(u.dacc.mean()),
                        dmase_pct=float(u.dmase_pct.mean())),
            ratio=float(u.dacc.mean() / s.dacc.mean()),
        )

    u = ok[ok.seen == 0].groupby(["run", "year"]).agg(
        dacc=("dacc", "mean"), dmase_pct=("dmase_pct", "mean")).reset_index()
    rho, p = stats.spearmanr(u.dacc, u.dmase_pct)
    tt = stats.ttest_1samp(u.dmase_pct, 0)
    out["run_level_unseen"] = dict(
        n=int(len(u)), spearman=float(rho), p=float(p),
        frac_score_up=float((u.dacc > 0).mean()),
        frac_error_worse=float((u.dmase_pct > 0).mean()),
        frac_tradeoff=float(((u.dacc > 0) & (u.dmase_pct > 0)).mean()),
        error_ttest_p=float(tt.pvalue),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = compute()

    print("C19 | %d run x farm observations, %d runs" % (res["n_rows"], res["n_runs"]))
    print("  %-6s %8s %6s %18s %18s" % ("year", "farms", "runs",
                                        "unseen farms dACC", "seen farms dACC"))
    for year in sorted(res["by_year"]):
        y = res["by_year"][year]
        print("  %-6d %8d %6d   %+.2f (n=%4d)      %+.2f (n=%3d)" % (
            year, y["n_farms"], y["n_runs"],
            y["unseen"]["dacc"], y["unseen"]["n"],
            y["seen"]["dacc"], y["seen"]["n"]))
    print("  error on unseen farms: " + ", ".join(
        "%+.2f%%" % res["by_year"][y]["unseen"]["dmase_pct"] for y in sorted(res["by_year"])))

    r = res["run_level_unseen"]
    print("\nRun level (unseen farms, n=%d)" % r["n"])
    print("  score up in %.1f%% of runs; error worse in %.1f%%; both in %.1f%%" % (
        100 * r["frac_score_up"], 100 * r["frac_error_worse"], 100 * r["frac_tradeoff"]))
    print("  Spearman(score gain, error change) = %.3f (p=%.2e); error t-test p=%.2e" % (
        r["spearman"], r["p"], r["error_ttest_p"]))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
