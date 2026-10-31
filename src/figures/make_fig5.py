# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
f = pd.read_csv(os.path.join(ROOT, "data", "tables", "c21_floor.csv"))
ho = pd.read_csv(os.path.join(ROOT, "data", "tables", "c21_ho.csv"))
c19 = pd.read_csv(os.path.join(ROOT, "data", "tables", "c19_station_ext.csv"))
low = pd.read_csv(os.path.join(ROOT, "data", "tables", "c24_lowshare.csv"))

fig = plt.figure(figsize=(7.1, 2.55))
gs = GridSpec(1, 3, width_ratios=[1.0, 1.1, 1.0], wspace=0.38,
              left=0.075, right=0.985, top=0.85, bottom=0.19)

# ---- a: grouped bars ----
ax = fig.add_subplot(gs[0, 0])
groups, lo, hi = [], [], []
for y in (2023, 2024, 2025):
    s = f[f.year == y]
    groups.append(str(y))
    lo.append(s[s.stratum == "low"].dacc.mean())
    hi.append(s[s.stratum == "high"].dacc.mean())
groups.append("holdout")
lo.append(ho[ho.stratum == "low"].dacc.mean())
hi.append(ho[ho.stratum == "high"].dacc.mean())
x = np.arange(len(groups)); w = 0.34
ax.bar(x - w/2, lo, w, color=S.VERM, lw=0, label="below floor")
ax.bar(x + w/2, hi, w, color=S.BLUE, lw=0, label="above floor")
ax.axhline(0, color="#666666", lw=0.6)
for xx, v in zip(x - w/2, lo):
    ax.text(xx, v + 0.12, "%.1f" % v, ha="center", fontsize=5.2, color=S.VERM)
for xx, v in zip(x + w/2, hi):
    ax.text(xx, v - 0.42, "%.1f" % v, ha="center", fontsize=5.2, color=S.BLUE)
ax.axvline(2.5, color="#CCCCCC", lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(-2.6, 6.2); ax.set_ylabel("Δ score within regime (points)")
ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.15, labelspacing=0.2, fontsize=5.8)
S.panel(ax, "a")

# ---- b: low share vs gain ----
ax2 = fig.add_subplot(gs[0, 1])
m = c19.merge(low[["year", "station", "low_share"]], on=["year", "station"], how="inner")
m = m[m.dacc.notna() & (m.seen == 0)]
YC = {2023: S.ORANGE, 2024: S.SKY, 2025: S.BLUE}
for y, c in YC.items():
    s = m[m.year == y]
    _pc = ax2.scatter(s.low_share * 100, s.dacc, s=9, color=c, alpha=0.55, lw=0, label=str(y))
    _pc.set_rasterized(True)
b = np.polyfit(m.low_share * 100, m.dacc, 1)
xx = np.linspace(42, 100, 30)
ax2.plot(xx, np.polyval(b, xx), color="#333333", lw=0.8)
r, p = stats.spearmanr(m.low_share, m.dacc)
ax2.axhline(0, color="#999999", lw=0.5)
ax2.set_xlabel("Intervals below 0.2·Cap on that farm (%)")
ax2.set_ylabel("Δ score on that farm (points)")
ax2.set_xlim(42, 100); ax2.set_ylim(-2.5, 7.4)
ax2.legend(loc="upper right", handletextpad=0.3, borderpad=0.15, labelspacing=0.2,
           fontsize=5.6, markerscale=1.4)
ax2.text(0.03, 0.97, "ρ = %.2f (p < 0.001)" % r, transform=ax2.transAxes, fontsize=6.0,
         va="top", ha="left")
S.panel(ax2, "b", x=-0.30)

# ---- c: within-run trade-off ----
ax3 = fig.add_subplot(gs[0, 2])
r2 = f.groupby(["run", "stratum"]).dacc.mean().unstack().reset_index().dropna()
ax3.axhline(0, color="#999999", lw=0.5); ax3.axvline(0, color="#999999", lw=0.5)
ax3.scatter(r2.low, r2.high, s=13, color=S.PURPLE, alpha=0.8, edgecolor="white", lw=0.35)
b2 = np.polyfit(r2.low, r2.high, 1)
xx2 = np.linspace(-1, 9, 20)
ax3.plot(xx2, np.polyval(b2, xx2), color="#333333", lw=0.8)
rr, pp = stats.spearmanr(r2.low, r2.high)
ax3.set_xlabel("Δ score below floor (points)")
ax3.set_ylabel("Δ score above floor (points)")
ax3.set_xlim(-1, 9); ax3.set_ylim(-5, 3)
ax3.text(0.03, 0.06, "ρ = %.2f" % rr, transform=ax3.transAxes, fontsize=6.0,
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
ax3.text(0.97, 0.06, "n = %d runs" % len(r2), transform=ax3.transAxes,
         ha="right", fontsize=5.6,
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
S.panel(ax3, "c", x=-0.32)

S.save(fig, "fig5_regime", FIG)
print("rho_b", round(r, 3), "rho_c", round(rr, 3), len(r2))
