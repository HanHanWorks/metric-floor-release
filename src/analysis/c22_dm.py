# -*- coding: utf-8 -*-
"""C22 | Diebold-Mariano tests on identical forecasts, under three losses.

Reads data/tables/c22_dm.csv: one row per run x farm x loss, holding the DM
statistic for the daily loss differential (baseline minus pipeline), with the
Newey-West long-run variance and the Harvey-Leybourne-Newbold small-sample
correction applied upstream.

Reproduces Table 7 and Section 4.4.

Usage:
    python src/analysis/c22_dm.py [--json results/analysis/c22.json]
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

LABEL = {"metric": "assessment rule", "se": "squared error", "ae": "absolute error"}


def compute():
    d = pd.read_csv(os.path.join(TABLES, "c22_dm.csv"))
    d = d[d.dm.notna()].copy()
    out = dict(n_tests=int(len(d)), by_loss={})
    for loss, g in d.groupby("loss"):
        unseen = g[g.seen == 0]
        seen = g[g.seen == 1]
        k = int((unseen.dm > 0).sum())
        out["by_loss"][loss] = dict(
            n_unseen=int(len(unseen)), n_seen=int(len(seen)),
            frac_positive=float((unseen.dm > 0).mean()),
            median_dm=float(unseen.dm.median()),
            frac_sig_better=float((unseen.dm > 1.96).mean()),
            frac_sig_worse=float((unseen.dm < -1.96).mean()),
            sign_test_k=k, sign_test_n=int(len(unseen)),
            sign_test_p=float(stats.binom_test(k, len(unseen), 0.5)),
            median_relative_gain=float(unseen.rel.median()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = compute()

    print("C22 | %d tests, %d unseen-farm tests per loss" % (
        res["n_tests"], res["by_loss"]["metric"]["n_unseen"]))
    print("  %-16s %9s %10s %9s %9s %11s %12s" % (
        "loss", "DM>0", "median DM", "sig better", "sig worse", "sign test", "p"))
    for loss in ("metric", "se", "ae"):
        r = res["by_loss"][loss]
        print("  %-16s %8.1f%% %+9.2f %8.1f%% %8.1f%%   %4d/%4d  %.1e" % (
            LABEL[loss], 100 * r["frac_positive"], r["median_dm"],
            100 * r["frac_sig_better"], 100 * r["frac_sig_worse"],
            r["sign_test_k"], r["sign_test_n"], r["sign_test_p"]))
    print("\n  positive statistics indicate the pipeline is better")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
