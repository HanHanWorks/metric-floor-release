# -*- coding: utf-8 -*-
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG  = os.path.join(ROOT, "figures")
RES  = os.path.join(ROOT, "data", "runs")
prev = pd.read_csv(os.path.join(ROOT, "data", "tables", "c23_mech_prevalence.csv"))
fl   = pd.read_csv(os.path.join(ROOT, "data", "tables", "c21_floor.csv"))

SHORT = {"M01_global_scale": "global rescaling", "M02_smoothing": "smoothing",
         "M03_median_filter": "median filter", "M04_spike_removal": "spike removal",
         "M05_floor": "low-output floor", "M06_low_shrink": "low-output shrink",
         "M07_clip_capacity": "capacity clip", "M08_ramp_limit": "ramp limit",
         "M09_lag_blend": "lag blend", "M10_quantile_stretch": "quantile stretch",
         "M11_calendar_correction": "calendar correction", "M12_piecewise": "piecewise"}

fig = plt.figure(figsize=(7.15, 2.75))
gs = GridSpec(1, 3, width_ratios=[1.05, 1.0, 1.08], wspace=0.62,
              left=0.155, right=0.985, top=0.85, bottom=0.17)

# ---- a: prevalence ----
ax = fig.add_subplot(gs[0, 0])
keys = list(SHORT.keys())
cnt = [int(prev.mechs.fillna("").str.contains(k, regex=False).sum()) for k in keys]
pct = [100 * c / len(prev) for c in cnt]
o = np.argsort(pct)
y = np.arange(len(o))
for j, i in enumerate(o):
    ax.plot([0, pct[i]], [j, j], color="#CBD5E1", lw=0.8, zorder=1)
    ax.scatter([pct[i]], [j], s=17, color=S.BLUE if keys[i] != "M05_floor" else S.VERM, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([SHORT[keys[i]] for i in o], fontsize=6.0)
for t, i in zip(ax.get_yticklabels(), o):
    if keys[i] == "M05_floor":
        t.set_color(S.VERM); t.set_fontweight("bold")
ax.set_xlim(0, 100); ax.set_ylim(-0.7, len(o) - 0.3)
ax.set_xlabel("% of 106 final pipelines")
for j, i in enumerate(o):
    ax.text(pct[i] + 2.5, j, "%d" % cnt[i], va="center", fontsize=5.4, color="#475569")

S.panel(ax, "a", x=-0.46)

# ---- b: marginal returns ----
ax2 = fig.add_subplot(gs[0, 1])
curves = []
for fp in sorted(glob.glob(RES + "/steps_*.csv")):
    d = pd.read_csv(fp).sort_values("it")
    acc = np.maximum.accumulate(d.acc.values)
    ac = d.accepted.values
    tot = acc[-1] - acc[0]
    if tot <= 1e-9:
        continue
    idx = np.where(ac == 1)[0]
    if len(idx) < 3:
        continue
    c = [(acc[i] - acc[0]) / tot for i in idx][:20]
    curves.append(c)
K = 20
mat = np.full((len(curves), K), np.nan)
for i, c in enumerate(curves):
    mat[i, :len(c)] = c
mean = np.nanmean(mat, 0); q1 = np.nanpercentile(mat, 25, 0); q3 = np.nanpercentile(mat, 75, 0)
kx = np.arange(1, K + 1)
ax2.fill_between(kx, q1, q3, color=S.BLUE, alpha=0.15, lw=0)
ax2.plot(kx, mean, color=S.BLUE, lw=1.0, marker="o", ms=2.4)
ax2.axhline(1, color="#999999", lw=0.5, ls=(0, (3, 2)))
ax2.axvline(10, color=S.VERM, lw=0.6, ls=(0, (3, 2)))
ax2.text(10.4, 0.08, "10th edit:\n%.0f%% of gain" % (100 * mean[9]), fontsize=5.6, color=S.VERM)
ax2.set_xlim(1, K); ax2.set_ylim(0, 1.12)
ax2.set_xticks([1, 5, 10, 15, 20])
ax2.set_xlabel("Accepted edit number"); ax2.set_ylabel("Cumulative share of run's gain")
ax2.text(0.03, 0.94, "n = %d runs (≥3 edits)" % len(curves), transform=ax2.transAxes, va="top", fontsize=5.6)
S.panel(ax2, "b", x=-0.30)

# ---- c: with/without forest ----
ax3 = fig.add_subplot(gs[0, 2])
m = fl.groupby(["run", "stratum"]).dacc.mean().unstack().reset_index().dropna()
m = m.merge(prev[["run", "mechs"]], on="run", how="inner")
m["mset"] = m.mechs.fillna("").apply(lambda s: set(x for x in s.split(";") if x))
rows = []
for k in keys:
    has = m[m.mset.apply(lambda s: k in s)]; no = m[~m.run.isin(has.run)]
    if len(has) < 8 or len(no) < 8:
        continue
    diff = has.low.mean() - no.low.mean()
    se = np.sqrt(has.low.var(ddof=1) / len(has) + no.low.var(ddof=1) / len(no))
    rows.append((SHORT[k], diff, diff - 1.96 * se, diff + 1.96 * se, len(has)))
rows.sort(key=lambda r: r[1])
yy = np.arange(len(rows))
for j, (nm, dv, lo, hi, n) in enumerate(rows):
    sig = (lo > 0) or (hi < 0)
    c = S.VERM if sig else "#94A3B8"
    ax3.plot([lo, hi], [j, j], color=c, lw=0.9)
    ax3.scatter([dv], [j], s=16, color=c, zorder=3)
    ax3.text(4.35, j, "%d" % n, va="center", ha="right", fontsize=5.0, color="#64748B")
ax3.axvline(0, color="#666666", lw=0.6)
ax3.set_yticks(yy); ax3.set_yticklabels([r[0] for r in rows], fontsize=6.0)
ax3.set_xlim(-4, 4.6); ax3.set_ylim(-0.7, len(rows) - 0.3)
ax3.set_xlabel("Δ floor-regime gain, with − without (points)")

S.panel(ax3, "c", x=-0.42)

S.save(fig, "fig6_what_changed", FIG)
print("runs curve", len(curves), "10th", round(mean[9], 3))
