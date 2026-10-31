# -*- coding: utf-8 -*-
"""统一论文图风格（matplotlib 2.2 兼容）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

BLUE   = "#0072B2"
VERM   = "#D55E00"
GREEN  = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY    = "#56B4E9"
GREY   = "#6B7280"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def panel(ax, s, x=-0.17, y=1.04, fs=8):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="bottom", ha="left", color="#111111")


def thin(ax, which="both"):
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def save(fig, name, root):
    fig.savefig(root + "/" + name + ".pdf")
    fig.savefig(root + "/" + name + ".png", dpi=400)
    print("saved", name)
