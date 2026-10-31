# -*- coding: utf-8 -*-
"""Tables built from the per-run artefacts in data/runs/.

Everything here is derivable from files that ship with this repository:
  meta_<run>.json   one record per run  (n_iter, n_accept, acc_first, acc_last)
  steps_<run>.csv   one row per candidate that produced a score
  code_<run>.py     the final accepted transform() for that run

Outputs (printed, and optionally written to results/analysis/):
  Table 4   corpus and score change by year
  Table 9   prevalence of the twelve mechanism classes
  Section 4.6  marginal returns and mechanism-group comparisons
  Section 4.5  acceptance counts

Usage:
    python src/analysis/trace_tables.py [--json results/analysis/trace_tables.json]
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "data", "runs")
TABLES = os.path.join(REPO, "data", "tables")

sys.path.insert(0, os.path.join(REPO, "src", "loop"))
from p3_ontology import ONTOLOGY, extract  # noqa: E402


# --- loaders ---------------------------------------------------------------

def load_meta():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RUNS, "meta_*.json"))):
        with open(fp, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    m = pd.DataFrame(rows)
    m["windowed"] = m["run"].str.contains("win")
    m["dacc"] = m["acc_last"] - m["acc_first"]
    return m


def load_steps():
    frames = []
    for fp in sorted(glob.glob(os.path.join(RUNS, "steps_*.csv"))):
        frames.append(pd.read_csv(fp))
    return pd.concat(frames, ignore_index=True)


def load_mechanisms():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RUNS, "code_*.py"))):
        run = os.path.basename(fp)[len("code_"):-len(".py")]
        code = open(fp, encoding="utf-8", errors="ignore").read()
        mech = extract(code)
        rows.append(dict(run=run, n_mech=len(mech), mechs=";".join(mech)))
    return pd.DataFrame(rows)


# --- analyses --------------------------------------------------------------

def table4(meta):
    """Corpus and score change by year (paper Table 4)."""
    g = meta.groupby("year")
    out = []
    for year, d in g:
        out.append(dict(
            year=int(year), runs=len(d),
            iterations=int(d.n_iter.sum()), accepted=int(d.n_accept.sum()),
            acceptance_rate=float(d.n_accept.sum()) / float(d.n_iter.sum()),
            dacc_mean=float(d.dacc.mean()), dacc_sd=float(d.dacc.std(ddof=1)),
        ))
    total = dict(year="total", runs=len(meta),
                 iterations=int(meta.n_iter.sum()),
                 accepted=int(meta.n_accept.sum()),
                 acceptance_rate=float(meta.n_accept.sum()) / float(meta.n_iter.sum()),
                 dacc_mean=float(meta.dacc.mean()),
                 dacc_sd=float(meta.dacc.std(ddof=1)))
    out.append(total)
    nw = meta[~meta.windowed]
    return dict(by_year=out,
                non_windowed=dict(runs=int(len(nw)),
                                  iterations=int(nw.n_iter.sum()),
                                  accepted=int(nw.n_accept.sum()),
                                  acceptance_rate=float(nw.n_accept.sum()) /
                                                  float(nw.n_iter.sum())),
                dacc_min=float(meta.dacc.min()),
                dacc_max=float(meta.dacc.max()),
                above_start=int((meta.dacc > 0).sum()),
                n_runs=int(len(meta)),
                n_windowed=int(meta.windowed.sum()))


def table9(mech):
    """Prevalence of the twelve mechanism classes (paper Table 9)."""
    n = len(mech)
    rows = []
    for name in ONTOLOGY:
        k = int(mech.mechs.str.split(";").apply(lambda s: name in s).sum())
        rows.append(dict(mechanism=name, pipelines=k, share=float(k) / n))
    rows.sort(key=lambda r: -r["pipelines"])
    return dict(n_pipelines=n, rows=rows,
                n_mech_mean=float(mech.n_mech.mean()),
                n_mech_min=int(mech.n_mech.min()),
                n_mech_max=int(mech.n_mech.max()))


def marginal_returns(steps):
    """Share of accepted edits that add no new mechanism; saturation curve."""
    acc = steps[steps.accepted == 1]
    n_acc = len(acc)
    redundant = int((acc.n_novel == 0).sum())

    share_first10 = []
    for run, d in steps.groupby("run"):
        d = d.sort_values("it")
        curve = np.maximum.accumulate(d.acc.values)
        total = curve[-1] - curve[0]
        idx = np.where(d.accepted.values == 1)[0]
        if len(idx) >= 10 and total > 0:
            share_first10.append((curve[idx[9]] - curve[0]) / total)

    return dict(n_accepted=n_acc,
                n_without_new_mechanism=redundant,
                share_without_new_mechanism=float(redundant) / n_acc,
                n_runs_ge10=int(len(share_first10)),
                share_gain_first10=float(np.mean(share_first10)))


def mechanism_groups(mech, floor):
    """Section 4.6: floor-regime gain by mechanism membership."""
    r = floor.groupby(["run", "stratum"]).dacc.mean().unstack()
    rows = []
    for _, m in mech.iterrows():
        if m["run"] not in r.index:
            continue
        row = dict(run=m["run"], low=float(r.low[m["run"]]),
                   high=float(r.high[m["run"]]), n_mech=int(m.n_mech))
        member = set(x for x in m.mechs.split(";") if x)
        for name in ONTOLOGY:
            row["has_" + name.split("_")[0]] = name in member
        rows.append(row)
    d = pd.DataFrame(rows)

    tests = []
    for name in ONTOLOGY:
        col = "has_" + name.split("_")[0]
        a, b = d[d[col]].low, d[~d[col]].low
        tests.append(dict(mechanism=name, n_with=int(len(a)),
                          mean_with=float(a.mean()), mean_without=float(b.mean()),
                          welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue)))
    rho, p = stats.spearmanr(d.n_mech, d.low)
    return dict(n_runs=int(len(d)), tests=tests,
                n_mechanism_classes_vs_floor_gain=dict(rho=float(rho), p=float(p)))


# --- entry point -----------------------------------------------------------

def compute():
    meta = load_meta()
    steps = load_steps()
    mech = load_mechanisms()
    floor = pd.read_csv(os.path.join(TABLES, "c21_floor.csv"))
    return dict(
        table4=table4(meta),
        table9=table9(mech),
        marginal=marginal_returns(steps),
        mechanism_groups=mechanism_groups(mech, floor),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    res = compute()

    t4 = res["table4"]
    print("Table 4 | corpus and score change by year")
    print("  %-6s %5s %10s %9s %9s %14s" %
          ("year", "runs", "iterations", "accepted", "rate", "mean dACC (SD)"))
    for row in t4["by_year"]:
        print("  %-6s %5d %10d %9d %8.1f%%      %+.2f (%.2f)" % (
            row["year"], row["runs"], row["iterations"], row["accepted"],
            100 * row["acceptance_rate"], row["dacc_mean"], row["dacc_sd"]))
    print("  range %.2f-%.2f; %d/%d runs above their own start" % (
        t4["dacc_min"], t4["dacc_max"], t4["above_start"], t4["n_runs"]))
    nw = t4["non_windowed"]
    print("  of which non-windowed: %d runs, %d iterations, %d accepted (%.1f%%)\n" % (
        nw["runs"], nw["iterations"], nw["accepted"], 100 * nw["acceptance_rate"]))

    t9 = res["table9"]
    print("Table 9 | mechanism prevalence (%d final pipelines, mean %.1f of 12 classes)"
          % (t9["n_pipelines"], t9["n_mech_mean"]))
    for row in t9["rows"]:
        print("  %-24s %3d  %5.1f%%" % (row["mechanism"], row["pipelines"], 100 * row["share"]))
    print("  classes per pipeline: range %d-%d\n" % (t9["n_mech_min"], t9["n_mech_max"]))

    m = res["marginal"]
    print("Marginal returns")
    print("  accepted edits without a new mechanism: %d/%d (%.1f%%)" % (
        m["n_without_new_mechanism"], m["n_accepted"],
        100 * m["share_without_new_mechanism"]))
    print("  first ten accepted edits carry %.0f%% of a run's gain (%d runs with >=10)\n" % (
        100 * m["share_gain_first10"], m["n_runs_ge10"]))

    g = res["mechanism_groups"]
    print("Floor-regime gain by mechanism membership (n=%d runs)" % g["n_runs"])
    for row in sorted(g["tests"], key=lambda r: r["welch_p"]):
        flag = "*" if row["welch_p"] < 0.05 else " "
        print("  %s %-24s %+.2f vs %+.2f  Welch p=%.4f  (n=%d)" % (
            flag, row["mechanism"], row["mean_with"], row["mean_without"],
            row["welch_p"], row["n_with"]))
    r = g["n_mechanism_classes_vs_floor_gain"]
    print("  classes per pipeline vs floor-regime gain: rho=%.3f p=%.4f" % (r["rho"], r["p"]))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
