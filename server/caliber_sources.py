# -*- coding: utf-8 -*-
"""Which forecast source reproduces the operator's annual table?

The archive carries two forecast series (`data_source` 1 and 2) and, for eleven
farms, a theory/available-power series. Our analysis uses source 1 throughout and
never tested the alternative, so the reconciliation gap reported in the paper
could be a source mismatch rather than a power-convention mismatch. This script
settles that for one year: it recomputes the annual score per farm under each
available source, joins the operator's table, and reports MAE, bias and
correlation for each.

Usage
    export WIND_DATA_ROOT=/path/to/archive
    python server/caliber_sources.py --year 2024 --out results/recon
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

R = os.environ.get("WIND_DATA_ROOT", "")
S = os.path.join(R, "2原始数据", "3Second_data")


def daily_acc(df, cap):
    df = df.dropna(subset=["PM", "PP"])
    if len(df) < 96 * 20:
        return np.nan, 0
    day = df.data_time.dt.normalize().values
    sq = ((df.PM.values - df.PP.values) /
          np.where(df.PM.values >= 0.2 * cap, df.PM.values, 0.2 * cap)) ** 2
    g = pd.DataFrame(dict(day=day, sq=sq)).groupby("day").agg(n=("sq", "size"), s=("sq", "sum"))
    g = g[g.n >= 90]
    if g.empty:
        return np.nan, 0
    return float((100 * (1 - np.sqrt(g.s / g.n))).mean()), int(len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="results/recon")
    a = ap.parse_args()
    if not R:
        sys.exit("WIND_DATA_ROOT is not set")

    cap = pd.read_csv(os.path.join(R, "4处理后数据", "filtered_56_farm.csv"))
    cap.columns = [c.strip().lstrip("\ufeff") for c in cap.columns]
    cap["farm_id"] = cap["farm_id"].astype(str).str.strip()
    cap["Cap"] = pd.to_numeric(cap["capability"].astype(str).str.strip(" '"), errors="coerce")
    cap["nm"] = cap["farm_name"].astype(str).str.strip(" '")
    cap = cap[(cap.Cap > 10) & cap.nm.str.contains("风电")]

    real = pd.read_csv(
        os.path.join(S, "75-电站实际功率%d-bs_data_real_fifteen_%d.csv" % (a.year, a.year)),
        usecols=["farm_id", "data_time", "real_power"], dtype={"farm_id": str},
        parse_dates=["data_time"])
    fc = pd.read_csv(
        os.path.join(S, "81-短期预测功率表%d-bs_data_short_power_%d.csv" % (a.year, a.year)),
        usecols=["farm_id", "data_time", "predict_power", "data_source"],
        dtype={"farm_id": str, "data_source": str}, parse_dates=["data_time"])
    for d in (real, fc):
        d["farm_id"] = d["farm_id"].str.strip()
    real["PM"] = pd.to_numeric(real.real_power, errors="coerce")
    fc["PP"] = pd.to_numeric(fc.predict_power, errors="coerce")
    real = real.drop(columns=["real_power"])
    fc = fc.drop(columns=["predict_power"]).drop_duplicates(["farm_id", "data_time", "data_source"])

    m = real.merge(fc, on=["farm_id", "data_time"], how="inner").merge(
        cap[["farm_id", "Cap", "nm"]], on="farm_id", how="inner")
    m = m[(m.PM >= 0) & m.PM.notna() & m.PP.notna()]
    print("[merge] %d rows, %d farms, sources %s"
          % (len(m), m.farm_id.nunique(), sorted(m.data_source.unique())))

    ref = pd.read_csv(os.path.join(R, "场站准确率明细表.csv"))
    ref.columns = [c.strip().lstrip("\ufeff") for c in ref.columns]
    ref = ref.rename(columns={"场站": "nm", "平均准确率(%)": "official"})
    ref["nm"] = ref.nm.astype(str).str.strip().str.replace("'", "")

    rows = []
    for src, g in m.groupby("data_source"):
        for fid, d in g.groupby("farm_id"):
            acc, nd = daily_acc(d.sort_values("data_time"), float(d.Cap.iloc[0]))
            if np.isfinite(acc):
                rows.append(dict(source=src, farm_id=fid, nm=str(d.nm.iloc[0]).replace("'", ""),
                                 acc=acc, n_days=nd))
    D = pd.DataFrame(rows).merge(ref[["nm", "official"]], on="nm", how="inner")
    D["diff"] = D.acc - D.official

    os.makedirs(a.out, exist_ok=True)
    D.to_csv(os.path.join(a.out, "caliber_sources_%d.csv" % a.year), index=False)

    print("\n%-8s %4s %8s %8s %7s %8s %8s" % ("source", "n", "MAE", "bias", "corr", "within1", "within3"))
    summary = {}
    for src, g in D.groupby("source"):
        ad = g["diff"].abs()
        r = float(np.corrcoef(g.acc, g.official)[0, 1]) if len(g) > 3 else float("nan")
        summary[str(src)] = dict(n=int(len(g)), mae=float(ad.mean()),
                                 bias=float(g["diff"].mean()), corr=r,
                                 within1=float((ad <= 1).mean()),
                                 within3=float((ad <= 3).mean()))
        print("%-8s %4d %8.2f %+8.2f %7.3f %7.0f%% %7.0f%%"
              % (src, len(g), ad.mean(), g["diff"].mean(), r,
                 100 * (ad <= 1).mean(), 100 * (ad <= 3).mean()))
    with open(os.path.join(a.out, "caliber_sources_%d.json" % a.year), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nmedian |diff| by source:")
    print(D.groupby("source")["diff"].apply(lambda x: x.abs().median()).to_string())


if __name__ == "__main__":
    main()
