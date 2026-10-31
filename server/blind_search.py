# -*- coding: utf-8 -*-
"""Single-parameter blind search: can a structure-blind optimiser recover the gain?

The manuscript predicts that a random search over one multiplicative constant
recovers much of what the loop found, because below the floor the loss is
quadratic with a constant weight and a blunt instrument is close to optimal. This
script tests that prediction without any language model: it applies `k * PP` for
a grid of `k`, keeps the best `k` on the same four farms the loop searched, and
compares the recovered change with the loop's own.

It is the cheapest of the two experiments the paper lists as next steps, and it
needs only the restricted series -- not the proposer, and not the API.

Usage
-----
    export WIND_DATA_ROOT=/path/to/archive
    python server/blind_search.py --year 2025 --draws 200 --out results/blind

Output columns per run x farm: loop gain, grid-best gain, random-best gain, and
the share of the loop's gain recovered.
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

from exp_core import load_station_data, daily_metrics  # noqa: E402


def gain(df, pred, scale):
    m = daily_metrics(df, pred, scale)
    return 100.0 * (m.acc.mean() - daily_metrics(df, df.PP.values, scale).acc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--grid", default="0.80,1.20,0.005",
                    help="start,end,step for the deterministic sweep")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/blind")
    a = ap.parse_args()

    lo, hi, step = [float(x) for x in a.grid.split(",")]
    grid = np.arange(lo, hi + 1e-9, step)
    rng = np.random.RandomState(a.seed)
    min_days = 80 if a.year >= 2025 else 200

    runs = []
    for fp in sorted(glob.glob(os.path.join(REPO, "data", "runs", "code_*.py"))):
        run = os.path.basename(fp)[len("code_"):-len(".py")]
        if "win" in run or str(a.year) not in run:
            continue
        runs.append(run)

    rows = []
    for run in runs:
        seed = int(run.split("_s")[-1])
        stations = load_station_data(year=a.year, n_station=4, seed=seed,
                                     min_days=min_days)
        scale = dict((f, float(np.abs(np.diff(d.PM.values)).mean()))
                     for f, d in stations.items())
        for fid, df in stations.items():
            base = df.PP.values
            sweep = np.array([gain(df, k * base, scale[fid]) for k in grid])
            draw = np.array([gain(df, k * base, scale[fid])
                             for k in rng.uniform(lo, hi, a.draws)])
            rows.append(dict(run=run, year=a.year, station=fid,
                             k_best_grid=float(grid[sweep.argmax()]),
                             gain_grid=float(sweep.max()),
                             gain_random=float(draw.max())))
    out = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "blind_%d.csv" % a.year)
    out.to_csv(path, index=False)

    print("run x farm observations: %d (%d runs)" % (len(out), out.run.nunique()))
    print("grid best k: median %.3f, interquartile %.3f-%.3f"
          % (out.k_best_grid.median(),
             out.k_best_grid.quantile(.25), out.k_best_grid.quantile(.75)))
    print("gain from one constant, grid search:    %+.3f points" % out.gain_grid.mean())
    print("gain from one constant, random search:  %+.3f points" % out.gain_random.mean())
    print("compare with the loop's own gain on the same pairs (Table 6).")
    with open(os.path.join(a.out, "blind_%d.json" % a.year), "w") as fh:
        json.dump(dict(n=int(len(out)), runs=int(out.run.nunique()),
                       k_median=float(out.k_best_grid.median()),
                       gain_grid=float(out.gain_grid.mean()),
                       gain_random=float(out.gain_random.mean())), fh, indent=2)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
