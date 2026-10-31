# -*- coding: utf-8 -*-
"""Checks added in revision, all computed from the released tables.

Three things a referee is entitled to ask, none of which the first submission
answered:

1. `farm_reuse()`   Is a farm labelled "unseen" unseen for one pipeline, or for
                    the whole study? Farms are drawn per run, so the two differ.
2. `clustered()`    The 960 farm x run pairs are not independent. Report standard
                    errors clustered on run and on farm, and the same contrast
                    computed with farms (or runs) as the unit of analysis.
3. `flow()`         Reconcile iterations attempted, candidates that failed to
                    execute, candidates scored, and edits accepted.

Usage:
    python src/analysis/revision_checks.py [--json results/analysis/revision.json]
"""
from __future__ import print_function

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "data", "runs")
TABLES = os.path.join(REPO, "data", "tables")


# --- 1. was the farm ever used in any search? ------------------------------

def farm_reuse():
    d = pd.read_csv(os.path.join(TABLES, "c19_station_ext.csv"))
    d = d[d.dacc.notna()].copy()

    searched = {}
    for (year, station), g in d[d.seen == 1].groupby(["year", "station"]):
        searched[(year, station)] = set(g.run.unique())

    unseen = d[d.seen == 0].copy()
    unseen["n_other_runs"] = [
        len(searched.get((r.year, r.station), set()) - {r.run}) for r in unseen.itertuples()]
    unseen["touched_elsewhere"] = unseen.n_other_runs > 0

    per_farm = (d.drop_duplicates(["year", "station"])
                  .assign(n_searches=lambda x: [len(searched.get((r.year, r.station), set()))
                                                for r in x.itertuples()]))

    a = unseen[~unseen.touched_elsewhere].dacc
    b = unseen[unseen.touched_elsewhere].dacc
    return dict(
        n_farm_years=int(len(per_farm)),
        n_never_searched=int((per_farm.n_searches == 0).sum()),
        searches_per_farm=dict(mean=float(per_farm.n_searches.mean()),
                               median=float(per_farm.n_searches.median()),
                               min=int(per_farm.n_searches.min()),
                               max=int(per_farm.n_searches.max())),
        unseen_pairs=int(len(unseen)),
        unseen_pairs_touched_elsewhere=int(unseen.touched_elsewhere.sum()),
        share_touched_elsewhere=float(unseen.touched_elsewhere.mean()),
        dacc_never_searched=dict(n=int(len(a)), mean=float(a.mean())),
        dacc_touched_elsewhere=dict(n=int(len(b)), mean=float(b.mean())),
        welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue),
        by_year={int(y): dict(
            n=int(len(g)),
            n_never=int((~g.touched_elsewhere).sum()),
            never=float(g[~g.touched_elsewhere].dacc.mean()) if (~g.touched_elsewhere).any() else None,
            touched=float(g[g.touched_elsewhere].dacc.mean()))
            for y, g in unseen.groupby("year")})


# --- 2. inference that keeps the run x farm structure ---------------------

def _two_way_se(df, col):
    """Standard error clustered on run, clustered on farm, and two-way."""
    v = {}
    for key in ("run", "station"):
        g = df.groupby(key)[col].agg(["sum", "count"])
        cluster_means = g["sum"] / g["count"]
        v[key] = cluster_means.var(ddof=1) / len(cluster_means)
    cell = df.groupby(["run", "station"])[col].mean()
    v["cell"] = cell.var(ddof=1) / len(cell)
    v["two_way"] = v["run"] + v["station"] - v["cell"]
    return dict((k, float(np.sqrt(max(x, 0.0)))) for k, x in v.items())


def clustered():
    d = pd.read_csv(os.path.join(TABLES, "c19_station_ext.csv"))
    d = d[d.dacc.notna()].copy()
    u = d[d.seen == 0].copy()
    u["dmase_pct"] = 100.0 * (u.mase_new - u.mase_base) / u.mase_base

    out = dict(n_pairs=int(len(u)), n_runs=int(u.run.nunique()),
               n_farms=int(u.station.nunique()), contrasts={})
    for col, label in (("dacc", "dacc"), ("dmase_pct", "dmase_pct")):
        est = float(u[col].mean())
        se = _two_way_se(u, col)
        naive = float(u[col].std(ddof=1) / np.sqrt(len(u)))
        out["contrasts"][label] = dict(
            estimate=est, se_naive=naive, se_run=se["run"], se_farm=se["station"],
            se_two_way=se["two_way"],
            ci_low=est - 1.96 * se["two_way"], ci_high=est + 1.96 * se["two_way"])

    dm = pd.read_csv(os.path.join(TABLES, "c22_dm.csv"))
    dm = dm[(dm.dm.notna()) & (dm.seen == 0)]
    out["dm_sign"] = {}
    for loss in ("metric", "se", "ae"):
        s = dm[dm.loss == loss]
        by_farm = s.groupby("station").apply(lambda g: (g.dm > 0).mean())
        by_run = s.groupby("run").apply(lambda g: (g.dm > 0).mean())
        t_farm = stats.ttest_1samp(by_farm, 0.5)
        t_run = stats.ttest_1samp(by_run, 0.5)
        out["dm_sign"][loss] = dict(
            pair_share=float((s.dm > 0).mean()), n_pairs=int(len(s)),
            farm_mean=float(by_farm.mean()), n_farms=int(len(by_farm)),
            farms_majority_positive=int((by_farm > 0.5).sum()),
            farm_t=float(t_farm.statistic), farm_p=float(t_farm.pvalue),
            run_mean=float(by_run.mean()), n_runs=int(len(by_run)),
            run_t=float(t_run.statistic), run_p=float(t_run.pvalue))
    return out


# --- 3. counting flow ------------------------------------------------------

def flow():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RUNS, "meta_*.json"))):
        with open(fp, encoding="utf-8") as fh:
            m = json.load(fh)
        s = pd.read_csv(os.path.join(RUNS, "steps_%s.csv" % m["run"]))
        rows.append(dict(run=m["run"], windowed="win" in m["run"],
                         planned=m["n_iter"], scored=len(s),
                         failed=m["n_iter"] - len(s),
                         last_it=int(s.it.max()), accepted=int(s.accepted.sum())))
    f = pd.DataFrame(rows)
    nw = f[~f.windowed]

    # proposals that carry a score differential
    diffs = []
    for fp in sorted(glob.glob(os.path.join(RUNS, "steps_*.csv"))):
        s = pd.read_csv(fp).sort_values("it")
        s["d_acc"] = s.acc.diff()
        diffs.append(s)
    dd = pd.concat(diffs, ignore_index=True)
    dd = dd[dd.d_acc.notna()]
    acc = dd[dd.accepted == 1]

    return dict(
        runs=int(len(f)), windowed=int(f.windowed.sum()),
        iterations_planned=int(f.planned.sum()),
        iterations_scored=int(f.scored.sum()),
        candidates_failed=int(f.failed.sum()),
        accepted=int(f.accepted.sum()),
        rejected=int(f.scored.sum() - f.accepted.sum()),
        acceptance_rate_all=float(f.accepted.sum()) / float(f.planned.sum()),
        acceptance_rate_scored=float(f.accepted.sum()) / float(f.scored.sum()),
        non_windowed=dict(runs=int(len(nw)),
                          reached_plan=int((nw.last_it >= 60).sum()),
                          stopped_early=int((nw.last_it < 60).sum()),
                          last_it_range=[int(nw.last_it.min()), int(nw.last_it.max())]),
        with_differential=dict(n=int(len(dd)), accepted=int(dd.accepted.sum()),
                               rejected=int((dd.accepted == 0).sum()),
                               accepted_mean=float(acc.d_acc.mean()),
                               accepted_sd=float(acc.d_acc.std(ddof=1)),
                               rejected_mean=float(dd[dd.accepted == 0].d_acc.mean()),
                               rejected_sd=float(dd[dd.accepted == 0].d_acc.std(ddof=1)),
                               non_positive_accepted=int((acc.d_acc <= 0).sum())))


def compute():
    return dict(farm_reuse=farm_reuse(), clustered=clustered(), flow=flow())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    r = compute()

    fr = r["farm_reuse"]
    print("1. Farm reuse")
    print("   %d farm-years, %d never used in any search; a farm is used in a median of %d runs (max %d)"
          % (fr["n_farm_years"], fr["n_never_searched"],
             fr["searches_per_farm"]["median"], fr["searches_per_farm"]["max"]))
    print("   of %d unseen pairs, %d (%.1f%%) involve a farm searched by another run"
          % (fr["unseen_pairs"], fr["unseen_pairs_touched_elsewhere"],
             100 * fr["share_touched_elsewhere"]))
    print("   dACC on never-searched farms %+.2f (n=%d) vs searched elsewhere %+.2f (n=%d), Welch p=%.2f"
          % (fr["dacc_never_searched"]["mean"], fr["dacc_never_searched"]["n"],
             fr["dacc_touched_elsewhere"]["mean"], fr["dacc_touched_elsewhere"]["n"],
             fr["welch_p"]))

    cl = r["clustered"]
    print("\n2. Clustered inference (%d pairs, %d runs, %d farms)"
          % (cl["n_pairs"], cl["n_runs"], cl["n_farms"]))
    for label in ("dacc", "dmase_pct"):
        x = cl["contrasts"][label]
        print("   %-9s %+.3f  SE naive %.3f, run %.3f, farm %.3f, two-way %.3f  95%% CI [%+.3f, %+.3f]"
              % (label, x["estimate"], x["se_naive"], x["se_run"], x["se_farm"],
                 x["se_two_way"], x["ci_low"], x["ci_high"]))
    for loss in ("metric", "se", "ae"):
        x = cl["dm_sign"][loss]
        print("   DM %-6s pairs %.1f%% | farm mean %.3f (t=%.2f, p=%.1e; %d/%d farms above half) "
              "| run mean %.3f (t=%.2f)" % (
                  loss, 100 * x["pair_share"], x["farm_mean"], x["farm_t"], x["farm_p"],
                  x["farms_majority_positive"], x["n_farms"], x["run_mean"], x["run_t"]))

    fl = r["flow"]
    print("\n3. Counting flow")
    print("   %d runs, %d iterations planned, %d scored, %d failed to execute"
          % (fl["runs"], fl["iterations_planned"], fl["iterations_scored"],
             fl["candidates_failed"]))
    print("   %d accepted (%.1f%% of attempted, %.1f%% of scored), %d rejected"
          % (fl["accepted"], 100 * fl["acceptance_rate_all"],
             100 * fl["acceptance_rate_scored"], fl["rejected"]))
    nw = fl["non_windowed"]
    print("   non-windowed: %d runs, %d reached iteration 60, %d stopped at %d-%d"
          % (nw["runs"], nw["reached_plan"], nw["stopped_early"],
             nw["last_it_range"][0], nw["last_it_range"][1]))
    w = fl["with_differential"]
    print("   proposals with a computable differential: %d (%d accepted %+.3f, %d rejected %+.3f); "
          "%d accepted had a non-positive change"
          % (w["n"], w["accepted"], w["accepted_mean"], w["rejected"], w["rejected_mean"],
             w["non_positive_accepted"]))

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
