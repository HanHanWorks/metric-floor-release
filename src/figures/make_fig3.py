# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
d = pd.read_csv(os.path.join(ROOT, "data", "tables", "c19_station_ext.csv"))
d = d[d.dacc.notna()].copy()
d["dmase_pct"] = 100 * (d.mase_new - d.mase_base) / d.mase_base

g = d.groupby(["run", "year", "seen"]).agg(dacc=("dacc", "mean"),
                                           dmase=("dmase_pct", "mean")).reset_index()
def side(flag):
    s = g[g.seen == flag].set_index(["run", "year"])
    return s.rename(columns={"dacc": "dacc", "dmase": "dmase"})
seen_ = side(1); unseen_ = side(0)
run = seen_.join(unseen_, lsuffix="_seen", rsuffix="_unseen", how="inner").reset_index().dropna()
assert len(run) == 56, len(run)
YC = {2023: S.ORANGE, 2024: S.SKY, 2025: S.BLUE}

fig = plt.figure(figsize=(7.1, 2.45))
gs = GridSpec(1, 3, width_ratios=[1, 1.08, 0.92], wspace=0.42,
              left=0.065, right=0.99, top=0.85, bottom=0.17)

# ---- a: seen vs unseen ----
ax = fig.add_subplot(gs[0, 0])
lim = (-0.4, 2.4)
ax.plot(lim, lim, color="#999999", lw=0.6, ls=(0, (3, 2)), zorder=0)
for y, c in YC.items():
    s = run[run.year == y]
    ax.scatter(s.dacc_seen, s.dacc_unseen, s=13, color=c, edgecolor="white",
               linewidth=0.35, label=str(y), zorder=3)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("Δ score on farms used in search"); ax.set_ylabel("Δ score on unseen farms")
ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.legend(loc="upper left", handletextpad=0.2, borderpad=0.15, labelspacing=0.2,
          markerscale=0.9, title="year", fontsize=5.8)
S.panel(ax, "a")

# ---- b: score gain vs MASE ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.add_patch(Rectangle((0, 0), 3, 6, color="#FDE7E7", lw=0, zorder=0))
ax2.add_patch(Rectangle((0, -6), 3, 6, color="#E7F6EC", lw=0, zorder=0))
ax2.axhline(0, color="#999999", lw=0.5); ax2.axvline(0, color="#999999", lw=0.5)
for y, c in YC.items():
    s = run[run.year == y]
    ax2.scatter(s.dacc_unseen, s.dmase_unseen, s=13, color=c, edgecolor="white",
                linewidth=0.35, zorder=3)
r, p = stats.spearmanr(run.dacc_unseen, run.dmase_unseen)
b = np.polyfit(run.dacc_unseen, run.dmase_unseen, 1)
xx = np.linspace(0, 2.2, 20)
ax2.plot(xx, np.polyval(b, xx), color="#333333", lw=0.7, ls="-", zorder=4)
ax2.set_xlim(0, 2.2); ax2.set_ylim(-4, 5.5)
ax2.set_xlabel("Δ score on unseen farms (points)")
ax2.set_ylabel("Δ MASE on unseen farms (%)")
ax2.text(0.04, 0.95, "ρ = %.2f\np = %.0e" % (r, p), transform=ax2.transAxes,
         va="top", ha="left", fontsize=6.2)
S.panel(ax2, "b")

# ---- c: ECDF ----
ax3 = fig.add_subplot(gs[0, 2])
for col, c, lab, ls in (("dacc_seen", S.VERM, "used in search", "-"),
                        ("dacc_unseen", S.BLUE, "unseen", "-")):
    v = np.sort(run[col].values)
    ax3.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=0.9, ls=ls, label=lab)
ax3.set_xlim(-0.2, 2.2); ax3.set_ylim(0, 1.02)
ax3.set_xlabel("Δ score (points)"); ax3.set_ylabel("Cumulative share of runs")
ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax3.legend(loc="lower right", handletextpad=0.4, borderpad=0.2, labelspacing=0.2)
ax3.text(0.03, 0.95, "n = 56 runs", transform=ax3.transAxes, va="top", fontsize=5.8)
S.panel(ax3, "c", x=-0.30)

S.save(fig, "fig3_cross_site", FIG)
print("rho", round(r, 3), "p", p)
