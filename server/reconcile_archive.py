# -*- coding: utf-8 -*-
"""Reconcile the whole archive against the operator's annual accuracy table.

The manuscript reports a four-farm spot check: two farms agree with the
operator's table to within a point, two differ by 21-23 points in opposite
directions. That is enough to raise the question but not to characterise it. This
script does the reconciliation over every farm in the register and every year in
the archive, and reports the distribution rather than two anecdotes.

It also tests the one convention whose direction is determinate. If the operator's
chain normalises by available rather than metered power, then for P_A >= P_M the
recomputed score can only fall relative to ours; farms where the official value is
*below* ours therefore cannot be explained by that convention.

Inputs  (defaults are the archive layout used by exp_core)
    --official  场站准确率明细表.csv    columns: 场站, 平均准确率(%)
    --years      2023,2024,2025

Output
    recon_archive.csv   farm_id, name, year, cap, n_days, acc_ours, acc_official, diff
    recon_summary.json  per year: n matched, median |diff|, share within 1 point,
                        and the farms whose difference exceeds 5 points

Usage
    export WIND_DATA_ROOT=/path/to/archive
    python server/reconcile_archive.py --out results/recon
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "loop"))

from exp_core import _load_raw  # noqa: E402


def annual_acc(m, min_days=60):
    """Mean of daily assessment scores, per farm, on the merged archive."""
    out = []
    for fid, d in m.groupby("farm_id"):
        d = d.sort_values("data_time")
        cap = float(d.Cap.iloc[0])
        denom = np.where(d.PM.values >= 0.2 * cap, d.PM.values, 0.2 * cap)
        sq = ((d.PM.values - d.PP.values) / denom) ** 2
        day = d.data_time.dt.normalize().values
        df = pd.DataFrame(dict(day=day, sq=sq))
        g = df.groupby("day").agg(n=("sq", "size"), s=("sq", "sum"))
        g = g[g.n >= 90]
        if len(g) < min_days:
            continue
        acc = 100.0 * (1.0 - np.sqrt(g.s / g.n))
        out.append(dict(farm_id=str(fid), name=str(d.nm.iloc[0]), cap=cap,
                        n_days=int(len(g)), acc_ours=float(acc.mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default="")
    ap.add_argument("--years", default="2023,2024,2025")
    ap.add_argument("--min-days", type=int, default=60)
    ap.add_argument("--out", default="results/recon")
    a = ap.parse_args()

    root = os.environ.get("WIND_DATA_ROOT", "")
    official_path = a.official or os.path.join(root, "场站准确率明细表.csv")
    if not os.path.exists(official_path):
        sys.exit("official table not found: %s" % official_path)

    off = pd.read_csv(official_path)
    off.columns = [c.strip().lstrip("\ufeff") for c in off.columns]
    name_col = [c for c in off.columns if "场站" in c][0]
    acc_col = [c for c in off.columns if "准确率" in c][0]
    off = off[[name_col, acc_col]].rename(columns={name_col: "name", acc_col: "acc_official"})
    off["name"] = off.name.astype(str).str.strip().str.replace("'", "")
    off["acc_official"] = pd.to_numeric(off.acc_official, errors="coerce")
    off = off.dropna()

    frames = []
    for year in [int(y) for y in a.years.split(",")]:
        print("[year] %d: loading archive" % year)
        m = _load_raw(year)
        ours = annual_acc(m, a.min_days)
        ours["year"] = year
        ours["name_key"] = ours.name.str.replace("'", "").str.strip()
        j = ours.merge(off, left_on="name_key", right_on="name", how="inner")
        j["diff"] = j.acc_ours - j.acc_official
        frames.append(j)
        del m
        print("       matched %d farms" % len(j))
    d = pd.concat(frames, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, "recon_archive.csv"), index=False)

    summary = {}
    for year, g in d.groupby("year"):
        ad = g["diff"].abs()
        summary[int(year)] = dict(
            n=int(len(g)),
            median_abs_diff=float(ad.median()),
            iqr_abs_diff=[float(ad.quantile(.25)), float(ad.quantile(.75))],
            share_within_1=float((ad <= 1.0).mean()),
            share_within_3=float((ad <= 3.0).mean()),
            corr=float(np.corrcoef(g.acc_ours, g.acc_official)[0, 1]),
            bias=float(g["diff"].mean()),
            outliers=[dict(name=r["name_key"], ours=round(r.acc_ours, 1),
                           official=round(r.acc_official, 1), diff=round(r["diff"], 1))
                      for _, r in g[ad > 5].iterrows()])
        print("[year] %d  n=%d  median|d|=%.2f  within 1pt %.0f%%  corr %.3f  bias %+.2f"
              % (year, len(g), summary[int(year)]["median_abs_diff"],
                 100 * summary[int(year)]["share_within_1"],
                 summary[int(year)]["corr"], summary[int(year)]["bias"]))
    with open(os.path.join(a.out, "recon_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("wrote %s" % os.path.join(a.out, "recon_archive.csv"))


if __name__ == "__main__":
    main()
