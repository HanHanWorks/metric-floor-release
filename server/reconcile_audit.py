# -*- coding: utf-8 -*-
"""Reconciliation audit: which input convention reproduces the operator's table?

The manuscript reports that our recomputed annual score agrees with the
operator's for two of the four farms in the main experiment and differs for the
other two by 21-23 points, in opposite directions, and argues from that sign
pattern that at most one of the two farms can be explained by an available-power
convention. This script is the audit that would identify the cause: it recomputes
each farm under every combination of

  * forecast source   -- each `data_source` present in the forecast table,
  * power convention  -- metered power, or available power in the denominator only,
                         or available power throughout,
  * capacity vintage  -- the static register, or the monthly series,

and reports the absolute difference from the official value for each cell. The
cell that reproduces the official number for a farm names the convention that
farm is settled under.

Inputs
------
--official  CSV with columns `farm_id,official_acc` (the operator's annual table;
            four rows for the main experiment, more if available).
--available CSV with columns `farm_id,data_time,available_power`, if the archive
            holds one. Optional; the corresponding variants are skipped without it.
--monthly   CSV with columns `farm_id,month,capability`, if monthly capacity
            exists. Optional.

This script needs the restricted series and was not executed as part of the
release, so treat the column names of the two optional files as arguments to
check rather than as verified facts.

Usage
-----
    export WIND_DATA_ROOT=/path/to/archive
    python server/reconcile_audit.py --year 2024 \
        --official official_2024.csv --available available.csv --monthly cap_monthly.csv \
        --out results/recon
"""
from __future__ import print_function

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "loop"))

from exp_core import R, S, daily_metrics  # noqa: E402


def load_merged(year):
    """Metered output, every forecast source, and the static capacity register."""
    if not R:
        sys.exit("WIND_DATA_ROOT is not set (see data/README.md).")
    cap = pd.read_csv(os.path.join(R, "4处理后数据", "filtered_56_farm.csv"))
    cap.columns = [c.strip().lstrip("\ufeff") for c in cap.columns]
    cap["farm_id"] = cap["farm_id"].astype(str).str.strip()
    cap["Cap"] = pd.to_numeric(cap["capability"].astype(str).str.strip(" '"), errors="coerce")
    cap["nm"] = cap["farm_name"].astype(str).str.strip(" '")
    cap = cap[(cap.Cap > 10) & cap.nm.str.contains("风电")]

    real = pd.read_csv(
        os.path.join(S, "75-电站实际功率%d-bs_data_real_fifteen_%d.csv" % (year, year)),
        usecols=["farm_id", "data_time", "real_power"], dtype={"farm_id": str},
        parse_dates=["data_time"])
    fc = pd.read_csv(
        os.path.join(S, "81-短期预测功率表%d-bs_data_short_power_%d.csv" % (year, year)),
        usecols=["farm_id", "data_time", "predict_power", "data_source"],
        dtype={"farm_id": str, "data_source": str}, parse_dates=["data_time"])

    for d in (real, fc):
        d["farm_id"] = d["farm_id"].str.strip()
    fc = fc.drop_duplicates(["farm_id", "data_time", "data_source"])

    m = real.merge(cap[["farm_id", "Cap", "nm"]], on="farm_id", how="inner")
    m["PM"] = pd.to_numeric(m.real_power, errors="coerce")
    m = m.dropna(subset=["PM"])
    m = m[m.PM >= 0]
    return m, fc, cap


def acc_of(df, pred):
    scale = float(np.abs(np.diff(df.PM.values)).mean())
    out = daily_metrics(df, pred, scale)
    return float(out.acc.mean()) if len(out) else np.nan


def variants(year, farm_id, merged, fc, cap, available, monthly):
    """Yield (label, score) for one farm."""
    base = merged[merged.farm_id == farm_id].copy()
    if base.empty:
        return
    sources = sorted(fc[fc.farm_id == farm_id].data_source.unique())

    for src in sources:
        f = fc[(fc.farm_id == farm_id) & (fc.data_source == src)]
        f = f.drop_duplicates(["farm_id", "data_time"])
        d = base.merge(f[["farm_id", "data_time", "predict_power"]],
                       on="farm_id", how="inner")
        d["PP"] = pd.to_numeric(d.predict_power, errors="coerce")
        d = d.dropna(subset=["PP"]).sort_values("data_time").reset_index(drop=True)
        if len(d) < 96 * 30:
            continue
        d["dt"] = d.data_time
        d["dayc"] = d.dt.dt.normalize().astype("int64") // 10 ** 9

        caps = {"static": d.Cap.copy()}
        if monthly is not None:
            mm = monthly[monthly.farm_id.astype(str) == str(farm_id)]
            if len(mm):
                caps["monthly"] = d.dt.dt.month.map(
                    dict(zip(mm.month.astype(int), mm.capability))).fillna(d.Cap)

        powers = {"metered": (d.PM.values, d.PM.values)}
        if available is not None:
            av = available[available.farm_id.astype(str) == str(farm_id)]
            if len(av):
                s = av.set_index("data_time").available_power
                pa = d.data_time.map(s)
                if pa.notna().mean() > 0.5:
                    pa = pa.fillna(d.PM).values
                    powers["available_denominator"] = (d.PM.values, pa)
                    powers["available_all"] = (pa, pa)

        for (cap_label, cap_vec), (pow_label, (PMv, PAv)) in itertools.product(
                caps.items(), powers.items()):
            for pp_label, pp in (("issued", d.PP.values),):
                x = d.copy()
                x["Cap"] = np.asarray(cap_vec, dtype=float)
                x["PM"] = PMv
                # available-power variants: normalise by PA where it exceeds PM
                den = np.maximum(PMv, 0.2 * x["Cap"].values)
                if pow_label != "metered":
                    den = np.maximum(PAv, 0.2 * x["Cap"].values)
                sq = ((x["PM"].values - pp) / den) ** 2
                days = x.dayc.values
                accs = []
                for day in np.unique(days):
                    s_ = sq[days == day]
                    if len(s_) >= 90:
                        accs.append(100 * (1 - np.sqrt(s_.mean())))
                if accs:
                    yield ("src=%s | power=%s | cap=%s" % (src, pow_label, cap_label),
                           float(np.mean(accs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--available", default="")
    ap.add_argument("--monthly", default="")
    ap.add_argument("--out", default="results/recon")
    a = ap.parse_args()

    merged, fc, cap = load_merged(a.year)
    off = pd.read_csv(a.official)
    off["farm_id"] = off.farm_id.astype(str)
    available = pd.read_csv(a.available, parse_dates=["data_time"]) if a.available else None
    monthly = pd.read_csv(a.monthly) if a.monthly else None
    if available is not None:
        available["farm_id"] = available.farm_id.astype(str)
    if monthly is not None:
        monthly["farm_id"] = monthly.farm_id.astype(str)

    rows = []
    for farm_id, official in zip(off.farm_id, off.official_acc):
        for label, acc in variants(a.year, farm_id, merged, fc, cap, available, monthly):
            rows.append(dict(farm_id=farm_id, variant=label, acc=acc,
                             official=official, abs_diff=abs(acc - official)))
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("no variant produced a score; check the paths and the year")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "recon_%d.csv" % a.year)
    out.to_csv(path, index=False)

    print("farm-level best variant (smallest |recomputed - official|):")
    for farm_id, g in out.groupby("farm_id"):
        best = g.loc[g.abs_diff.idxmin()]
        print("  %s  official %.1f  best %.1f  diff %+.1f   %s"
              % (farm_id, best.official, best.acc, best.acc - best.official, best.variant))
    print("\nvariants that fall within 1 point of the official value:")
    close = out[out.abs_diff <= 1.0]
    if close.empty:
        print("  none")
    else:
        for (variant,), g in close.groupby(["variant"]):
            print("  %-58s %d/%d farms" % (variant, len(g), off.shape[0]))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
