# -*- coding: utf-8 -*-
"""Analyse the floor-free counterfactual against the floored corpus.

The counterfactual ran the same loop -- same arms, same seeds, same farms, same
four search farms, same prompts -- with one change: the objective's denominator
is the installed capacity instead of max(P_M, 0.2 Cap), so there is no floor and
no region of constant weight. Runs write to ~/p3_nf/results with tags
NFY2025_<arm>_base_s<seed>.

This script reports the three things the paper's prediction names:

  1. How much gain does the search find when the floor is gone?
  2. Does the composition change -- is the constant-shift instrument still
     dominant, is the floor-naming operator still rare?
  3. When the floor-free pipelines are re-scored under the *floored* rule, do
     they still gain? (If they do not, the loop's gain was floor-specific.)

Usage (on the node with the archive):
    python server/nofloor_analysis.py --floored ~/hzh_RSI/04_experiments/P3_knowledge_retention/results \
        --nf ~/p3_nf/results --out results/nf
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "loop"))

from p3_ontology import ONTOLOGY, extract  # noqa: E402
from exp_core import load_station_data, daily_metrics  # noqa: E402

YEAR = 2025
ARMS = ("nomem", "full", "none", "regime")


def load_runs(results_dir, prefix=""):
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "meta_%s*.json" % prefix))):
        with open(fp, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["dacc"] = d.acc_last - d.acc_first
    return d


def arm_of(run):
    for a in ARMS:
        if a in run:
            return a
    return "other"


def prevalence(results_dir, prefix=""):
    rows = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "code_%s*.py" % prefix))):
        run = os.path.basename(fp)[len("code_"):-len(".py")]
        mech = extract(open(fp, encoding="utf-8", errors="ignore").read())
        rows.append(dict(run=run, arm=arm_of(run), n_mech=len(mech),
                         mechs=";".join(mech)))
    d = pd.DataFrame(rows)
    n = len(d)
    prev = {}
    for name in ONTOLOGY:
        k = int(d.mechs.str.split(";").apply(lambda s: name in s).sum())
        prev[name] = 100.0 * k / max(n, 1)
    return d, prev, float(d.n_mech.mean()) if n else np.nan


def rescore_under_floor(nf_dir, code_files):
    """Re-score each floor-free final pipeline under the floored rule."""
    import re as _re
    sys.path.insert(0, os.path.join(REPO, "src", "loop"))
    os.environ["P3_DENOM"] = "floor"
    g = {"np": np, "pd": pd, "len": len, "float": float, "int": int, "abs": abs,
         "min": min, "max": max, "range": range, "sum": sum, "enumerate": enumerate}
    rows = []
    for fp in code_files:
        run = os.path.basename(fp)[len("code_"):-len(".py")]
        m = _re.search(r"_s(\d+)$", run)
        if not m:
            continue
        seed = int(m.group(1))
        st = load_station_data(year=YEAR, n_station=4, seed=seed, min_days=80)
        code = open(fp, encoding="utf-8", errors="ignore").read()
        loc = {}
        exec(compile(code, "<nf>", "exec"), dict(g), loc)
        fn = loc.get("transform")
        if fn is None:
            continue
        base, new = [], []
        for fid, df in st.items():
            sc = float(np.abs(np.diff(df.PM.values)).mean())
            base.append(daily_metrics(df, df.PP.values, sc).acc.mean())
            v = df[["dt", "PP", "Cap"]].copy()
            v["hour"] = v.dt.dt.hour + v.dt.dt.minute / 60.0
            v["dow"] = v.dt.dt.dayofweek
            v["month"] = v.dt.dt.month
            try:
                pred = np.asarray(fn(v), dtype=float).ravel()
                new.append(daily_metrics(df, pred, sc).acc.mean())
            except Exception:
                new = []
                break
        if new:
            rows.append(dict(run=run, dacc_floored=float(np.mean(new) - np.mean(base))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floored", required=True)
    ap.add_argument("--nf", required=True)
    ap.add_argument("--out", default="results/nf")
    a = ap.parse_args()

    floored = load_runs(a.floored)
    floored = floored[floored.run.str.match(r"^Y%d_" % YEAR)
                      & ~floored.run.str.contains("win|ho")]
    nf = load_runs(a.nf, prefix="NF")

    print("1. Gain under the optimised objective (%d runs, year %d)" % (len(nf), YEAR))
    print("   floored corpus : %+.3f points (n=%d)" % (floored.dacc.mean(), len(floored)))
    print("   floor-free runs: %+.3f points (n=%d)" % (nf.dacc.mean(), len(nf)))
    by_arm = nf.groupby(nf.run.apply(arm_of)).dacc.mean()
    for arm in ARMS:
        if arm in by_arm.index:
            f = floored[floored.run.str.contains("_%s_" % arm) |
                        floored.run.str.match(r"^Y%d_%s_" % (YEAR, arm))]
            print("     %-7s nf %+.3f (n=%d) | floored %+.3f (n=%d)"
                  % (arm, by_arm[arm], int((nf.run.apply(arm_of) == arm).sum()),
                     f.dacc.mean() if len(f) else float("nan"), len(f)))

    d, prev, mean_mech = prevalence(a.nf, prefix="NF")
    d0, prev0, mean_mech0 = prevalence(a.floored)
    d0 = d0[d0.run.str.match(r"^Y%d_" % YEAR) & ~d0.run.str.contains("win|ho")]
    print("\n2. Composition (%d floor-free pipelines vs %d floored)" % (len(d), len(d0)))
    for name in ONTOLOGY:
        print("   %-24s floor-free %5.1f%%   floored %5.1f%%"
              % (name, prev[name], 100 * d0.mechs.str.split(";")
                 .apply(lambda s: name in s).mean()))
    print("   mechanism classes per pipeline: %.2f vs %.2f" % (mean_mech, mean_mech0))

    print("\n3. Floor-free pipelines re-scored under the floored rule")
    rf = rescore_under_floor(a.nf, sorted(glob.glob(os.path.join(a.nf, "code_*.py"))))
    os.makedirs(a.out, exist_ok=True)
    rf.to_csv(os.path.join(a.out, "nf_rescored.csv"), index=False)
    print("   mean gain under the floored rule: %+.3f points (n=%d runs)"
          % (rf.dacc_floored.mean(), len(rf)))
    with open(os.path.join(a.out, "nf_summary.json"), "w") as fh:
        json.dump(dict(nf_gain=float(nf.dacc.mean()), n_nf=int(len(nf)),
                       floored_gain=float(floored.dacc.mean()), n_floored=int(len(floored)),
                       rescored_gain=float(rf.dacc_floored.mean()), n_rescored=int(len(rf)),
                       prevalence_nf=prev, prevalence_floored={
                           k: 100.0 * d0.mechs.str.split(";").apply(
                               lambda s: k in s).mean() for k in ONTOLOGY}), fh, indent=2)
    print("wrote %s" % os.path.join(a.out, "nf_summary.json"))


if __name__ == "__main__":
    main()
