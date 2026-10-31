# -*- coding: utf-8 -*-
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import fig_style as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "data", "runs")
FIG  = os.path.join(ROOT, "figures")

# ---------- data ----------
tr = {}
fin_all = []
for fp in sorted(glob.glob(RES + "/steps_*.csv")):
    run = os.path.basename(fp)[6:-4]
    d = pd.read_csv(fp).sort_values("it")
    acc = np.maximum.accumulate(d.acc.values)
    fin_all.append(acc[-1] - acc[0])
    if ("win" not in run) and (len(acc) >= 20):
        tr[run] = acc[:60] - acc[0]
n_iter = 60
it = np.arange(1, n_iter + 1)
M = np.full((len(tr), n_iter), np.nan)
for i, v in enumerate(tr.values()):
    M[i, :len(v)] = v
n_t = np.sum(~np.isnan(M), 0)
mu = np.nanmean(M, 0)
sem = np.nanstd(M, 0, ddof=1) / np.sqrt(n_t)

ho = pd.read_csv(os.path.join(ROOT, "data", "tables", "c20_holdout.csv"))

fig = plt.figure(figsize=(7.1, 2.55))
gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.33,
              left=0.075, right=0.985, top=0.86, bottom=0.16)

# ---------- a: trajectory ----------
ax = fig.add_subplot(gs[0, 0])
for k in list(tr.keys())[:9]:
    v = tr[k]
    ax.plot(np.arange(1, len(v) + 1), v, color=S.GREY, lw=0.35, alpha=0.45, zorder=1)
ax.fill_between(it, mu - 1.96 * sem, mu + 1.96 * sem, color=S.BLUE, alpha=0.18, lw=0, zorder=2)
ax.plot(it, mu, color=S.BLUE, lw=1.1, zorder=3)
ax.axhline(0, color="#999999", lw=0.5, ls=(0, (3, 2)), zorder=0)
ax.set_xlim(1, n_iter); ax.set_ylim(-1.2, 2.4)
ax.text(0.985, 0.40, "n = %d → %d" % (n_t[0], n_t[-1]), transform=ax.transAxes,
        ha="right", fontsize=5.6, color="#64748B")
ax.set_xlabel("Iteration"); ax.set_ylabel("Cumulative change in score (points)")

ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
S.panel(ax, "a")

axi = inset_axes(ax, width="33%", height="31%", loc="upper left", borderpad=0.75)
fin = np.asarray(fin_all)
axi.hist(fin, bins=np.arange(0, 7.5, 0.5), color=S.BLUE, alpha=0.75, lw=0)
axi.axvline(fin.mean(), color=S.VERM, lw=0.8)
axi.set_yticks([]); axi.set_xlabel("Final change", fontsize=5.6, labelpad=1)
axi.tick_params(labelsize=5.2, length=1.8)
axi.spines["top"].set_visible(False); axi.spines["right"].set_visible(False)
axi.text(0.96, 0.84, "n = 106 runs", transform=axi.transAxes, ha="right", fontsize=5.4)

# ---------- b: holdout ----------
ax2 = fig.add_subplot(gs[0, 1])
lim = (-2.0, 6.2)
ax2.plot(lim, lim, color="#999999", lw=0.6, ls=(0, (3, 2)), zorder=0)
ax2.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
ax2.axvline(0, color="#BBBBBB", lw=0.5, zorder=0)
mk = {"station": ("o", S.BLUE, "Farms held out (n=16)"),
      "time":    ("s", S.VERM, "Days held out (n=16)")}
for mode, (m, c, lab) in mk.items():
    g = ho[ho["mode"] == mode]
    ax2.scatter(g.d_val, g.d_test, s=15, marker=m, facecolor=c, edgecolor="white",
                linewidth=0.4, alpha=0.95, label=lab, zorder=3)
for mode, c in (("station", S.BLUE), ("time", S.VERM)):
    g = ho[ho["mode"] == mode]
    ax2.plot([g.d_val.mean()], [g.d_test.mean()], marker="X", ms=6.5, color=c,
             mec="white", mew=0.5, zorder=5)
ax2.set_xlim(*lim); ax2.set_ylim(-2.0, 6.2)
ax2.set_xlabel("Change on acceptance subset (points)")
ax2.set_ylabel("Change on held-out subset (points)")
ax2.legend(loc="upper left", handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

_p = ax2.get_position()
axb = fig.add_axes([_p.x0 + 0.64 * _p.width, _p.y0 + 0.14 * _p.height,
                    0.30 * _p.width, 0.24 * _p.height])
vals = [[ho[ho["mode"]=="station"].d_val.mean(), ho[ho["mode"]=="station"].d_test.mean()],
        [ho[ho["mode"]=="time"].d_val.mean(),    ho[ho["mode"]=="time"].d_test.mean()]]
xs = np.array([0, 1]); w = 0.34
for i, (v, c) in enumerate(zip(vals, [S.BLUE, S.VERM])):
    b = axb.bar(xs - w/2 + i*w, v, w, color=c, lw=0, alpha=0.9)
    for xx, vv in zip(xs - w/2 + i*w, v):
        axb.text(xx, vv + 0.06, "%.1f" % vv, ha="center", va="bottom", fontsize=4.6)
axb.set_xticks(xs); axb.set_xticklabels(["farms", "days"], fontsize=5.2)
axb.tick_params(labelsize=5.2, length=1.6, pad=1)
axb.set_ylim(0, 2.6); axb.set_yticks([0, 1, 2])
axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
axb.set_title("mean change", fontsize=5.4, pad=1.5)
S.panel(ax2, "b", x=-0.24)

S.save(fig, "fig2_score_and_holdout", FIG)
