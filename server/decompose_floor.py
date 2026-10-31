# -*- coding: utf-8 -*-
"""Exact decomposition of the change in the assessment score by region.

Why this script exists
----------------------
The manuscript reports two *region sub-scores* (`+3.11` inside the floor regime,
`-0.91` outside it) and states explicitly that they do not partition the total
score. A referee is entitled to ask for the decomposition that does. It cannot be
computed from the released aggregate tables, because the rule averages a
within-day root mean square and the aggregate tables no longer contain the
per-day, per-region mean squared errors. This script recomputes them from the
restricted series and returns the exact decomposition.

The algebra
-----------
Write the daily score as

    ACC_d = 1 - D_d,     D_d = sqrt(w_Fd * A_Fd + w_Pd * A_Pd)

where A_Rd is the mean squared normalised error inside region R on day d, and
w_Rd = n_Rd / n_d its share of intervals. The change in the score is

    dACC = mean_d(D0_d) - mean_d(D1_d).

Because x - y = (x^2 - y^2)/(x + y), the region contributions

    c_R = mean_d[ w_Rd * (A_R0d - A_R1d) / (D0_d + D1_d) ]

sum exactly to dACC:

    c_F + c_P = dACC   (identically, not approximately).

The shares c_R / dACC are what Table 8 cannot give.

Usage (on the node that holds the metered and forecast series)
-------------------------------------------------------------
    export WIND_DATA_ROOT=/path/to/archive
    export P3_RESULTS=./results
    python server/decompose_floor.py --year 2025 --tag nomem --out results/decomp
"""
from __future__ import print_function

import argparse
import glob
import re
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "loop"))

from exp_core import load_station_data  # noqa: E402

RUN_PATTERN = r"^Y%d_(nomem|mem|full|none|regime)_s\d+$"


def make_view(df):
    v = df[["dt", "PP", "Cap"]].copy()
    v["hour"] = v.dt.dt.hour + v.dt.dt.minute / 60.0
    v["dow"] = v.dt.dt.dayofweek
    v["month"] = v.dt.dt.month
    return v


def run_code(code, view):
    g = {"np": np, "pd": pd, "len": len, "float": float, "int": int, "abs": abs,
         "min": min, "max": max, "range": range, "sum": sum, "enumerate": enumerate}
    loc = {}
    exec(compile(code, "<candidate>", "exec"), g, loc)
    return np.asarray(loc["transform"](view), dtype=float).ravel()


def day_region_stats(df, pred):
    """Per day: interval counts and mean squared normalised error, by region."""
    PM = df.PM.values.astype(float)
    PP = np.asarray(pred, dtype=float).ravel()
    C = float(df.Cap.iloc[0])
    dc = df.dayc.values.astype(np.int64)

    denom = np.where(PM >= 0.2 * C, PM, 0.2 * C)
    sq = ((PM - PP) / denom) ** 2
    floor = PM < 0.2 * C

    rows = []
    for day in np.unique(dc):
        m = dc == day
        n = int(m.sum())
        if n < 90:
            continue
        nF = int((m & floor).sum())
        nP = n - nF
        A_F = float(sq[m & floor].mean()) if nF else 0.0
        A_P = float(sq[m & ~floor].mean()) if nP else 0.0
        rows.append(dict(day=int(day), n=n, nF=nF, nP=nP, A_F=A_F, A_P=A_P))
    return pd.DataFrame(rows)


def decompose(run, year, stations, code):
    per_pair = []
    for fid, df in stations.items():
        view = make_view(df)
        base = day_region_stats(df, df.PP.values)
        new = day_region_stats(df, run_code(code, view))
        m = base.merge(new, on="day", suffixes=("_0", "_1"))
        if m.empty:
            continue
        wF0 = m.nF_0 / m.n_0
        wF1 = m.nF_1 / m.n_1
        wP0 = 1.0 - wF0
        wP1 = 1.0 - wF1
        D0 = np.sqrt(wF0 * m.A_F_0 + wP0 * m.A_P_0)
        D1 = np.sqrt(wF1 * m.A_F_1 + wP1 * m.A_P_1)
        gap = D0 + D1
        cF = float(np.mean(wF0 * (m.A_F_0 - m.A_F_1) / gap))
        cP = float(np.mean(wP0 * (m.A_P_0 - m.A_P_1) / gap))
        dacc = float(np.mean(D0) - np.mean(D1))
        per_pair.append(dict(
            run=run, year=year, station=fid, n_days=len(m),
            dacc=100 * dacc, c_floor=100 * cF, c_prop=100 * cP,
            share_floor=cF / (abs(cF) + abs(cP)) if (cF or cP) else np.nan,
            identity_error=100 * (cF + cP - dacc)))
    return pd.DataFrame(per_pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--tag", default="", help="restrict to runs containing this tag")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="farm inclusion threshold in days (default 200, 80 for 2025)")
    ap.add_argument("--all-farms", action="store_true",
                    help="evaluate every eligible farm in the year, not only the "
                         "four a run searched; this reproduces the 1,184-pair sample "
                         "of Table 6")
    ap.add_argument("--out", default="results/decomp")
    a = ap.parse_args()

    min_days = 80 if a.year >= 2025 else 200
    if a.tolerance:
        min_days = int(a.tolerance)

    if a.all_farms:
        # one load per year: the farm set does not depend on the run, only the
        # number of farms does, and n_station above the fleet size returns all.
        stations = load_station_data(year=a.year, n_station=999, seed=0,
                                     min_days=min_days)
        print("all-farms mode: %d farms in %d" % (len(stations), a.year))

    frames = []
    for fp in sorted(glob.glob(os.path.join(REPO, "data", "runs", "code_*.py"))):
        run = os.path.basename(fp)[len("code_"):-len(".py")]
        if not re.match(RUN_PATTERN % a.year, run):
            continue          # windowed, holdout and other campaigns are excluded
        if a.tag and a.tag not in run:
            continue
        if not a.all_farms:
            seed = int(run.split("_s")[-1])
            stations = load_station_data(year=a.year, n_station=4, seed=seed,
                                         min_days=min_days)
        code = open(fp, encoding="utf-8", errors="ignore").read()
        frames.append(decompose(run, a.year, stations, code))

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "decomp_%d.csv" % a.year)
    out.to_csv(path, index=False)

    print("pairs: %d" % len(out))
    print("max |identity error|: %.2e points (should be floating-point noise)"
          % out.identity_error.abs().max())
    print("mean dACC  %+.3f  =  floor %+.3f  +  proportional %+.3f"
          % (out.dacc.mean(), out.c_floor.mean(), out.c_prop.mean()))
    print("signed share of the gross movement inside the floor regime: %.1f%%"
          % (100 * out.share_floor.mean()))
    print("wrote %s" % path)
    with open(os.path.join(a.out, "decomp_%d.json" % a.year), "w") as fh:
        json.dump(dict(n_pairs=int(len(out)),
                       max_identity_error=float(out.identity_error.abs().max()),
                       mean=dict(dacc=float(out.dacc.mean()),
                                 c_floor=float(out.c_floor.mean()),
                                 c_prop=float(out.c_prop.mean()),
                                 share_floor=float(out.share_floor.mean()))),
                  fh, indent=2)


if __name__ == "__main__":
    main()
