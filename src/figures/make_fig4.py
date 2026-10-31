# -*- coding: utf-8 -*-
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
d = pd.read_csv(os.path.join(ROOT, "data", "tables", "c22_dm.csv"))
d = d[d.dm.notna() & (d.seen == 0)].copy()
ORDER = ["metric", "se", "ae"]
LBL = {"metric": "assessment\nloss", "se": "squared\nerror", "ae": "absolute\nerror"}
COL = {"metric": S.BLUE, "se": S.VERM, "ae": S.GREY}

fig = plt.figure(figsize=(7.1, 2.45))
gs = GridSpec(1, 3, width_ratios=[1.18, 1.0, 0.95], wspace=0.46,
              left=0.075, right=0.99, top=0.85, bottom=0.20)

# ---- a ----
ax = fig.add_subplot(gs[0, 0])
np.random.seed(0)
for i, k in enumerate(ORDER):
    v = d[d.loss == k].dm.values
    v = v[np.abs(v) < 12]
    _pc = ax.scatter(np.random.normal(i, 0.075, len(v)), v, s=1.5, color=COL[k],
                     alpha=0.16, lw=0, zorder=1)
    _pc.set_rasterized(True)
    bp = ax.boxplot([v], positions=[i], widths=0.40, patch_artist=True,
                    showfliers=False, zorder=3)
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor=COL[k], linewidth=0.7)
    for m in bp["medians"]:
        m.set(color=COL[k], linewidth=1.1)
ax.axhline(0, color="#999999", lw=0.5)
ax.axhline(1.96, color="#15803D", lw=0.5, ls=(0, (4, 2)))
ax.axhline(-1.96, color="#B91C1C", lw=0.5, ls=(0, (4, 2)))
ax.text(2.46, 2.05, "+1.96", fontsize=5.2, color="#15803D", ha="right")
ax.text(2.46, -2.55, "−1.96", fontsize=5.2, color="#B91C1C", ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([LBL[k] for k in ORDER])
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-8, 9)
ax.set_ylabel("Diebold–Mariano statistic\n(> 0: pipeline better)")

S.panel(ax, "a")

# ---- b: dumbbell ----
ax2 = fig.add_subplot(gs[0, 1])
better = np.array([100 * (d[d.loss == k].dm > 1.96).mean() for k in ORDER])
worse = np.array([100 * (d[d.loss == k].dm < -1.96).mean() for k in ORDER])
for i, k in enumerate(ORDER):
    ax2.plot([-worse[i], better[i]], [i, i], color=COL[k], lw=1.3, alpha=0.55, zorder=2)
    ax2.scatter([better[i]], [i], s=26, color=COL[k], edgecolor="white", lw=0.5, zorder=3)
    ax2.scatter([-worse[i]], [i], s=26, facecolor="white", edgecolor=COL[k], lw=0.9, zorder=3)
    ax2.text(better[i] + 2.5, i + 0.16, "%.0f" % better[i], fontsize=5.4, color=COL[k], va="center")
    ax2.text(-worse[i] - 2.5, i + 0.16, "%.0f" % worse[i], fontsize=5.4, color=COL[k],
             va="center", ha="right")
ax2.axvline(0, color="#666666", lw=0.6)
ax2.set_yticks(range(3)); ax2.set_yticklabels([LBL[k] for k in ORDER])
ax2.set_ylim(-0.6, 2.75); ax2.set_xlim(-52, 88)
ax2.set_xlabel("% of tests significant at 5%")
ax2.text(-46, 2.55, "← worse", fontsize=5.6, color="#B91C1C")
ax2.text(82, 2.55, "better →", fontsize=5.6, color="#15803D", ha="right")
S.panel(ax2, "b", x=-0.34)

# ---- c: by year, lines ----
ax3 = fig.add_subplot(gs[0, 2])
yrs = np.array([2023, 2024, 2025])
for k in ORDER:
    vals = [100 * (d[(d.loss == k) & (d.year == y)].dm > 0).mean() for y in yrs]
    ax3.plot(yrs, vals, marker="o", ms=3.0, lw=0.9, color=COL[k],
             label={"metric": "assessment", "se": "squared", "ae": "absolute"}[k])
ax3.axhline(50, color="#666666", lw=0.6, ls=(0, (3, 2)))
ax3.text(2025.25, 51.5, "50%", fontsize=5.2, color="#666666", ha="right")
ax3.set_xticks(yrs); ax3.set_xlim(2022.7, 2025.3); ax3.set_ylim(0, 105)
ax3.set_ylabel("% of tests with DM > 0")
ax3.legend(loc="lower left", handletextpad=0.4, borderpad=0.15, labelspacing=0.2, fontsize=5.6)
S.panel(ax3, "c", x=-0.36)

S.save(fig, "fig4_two_losses", FIG)
