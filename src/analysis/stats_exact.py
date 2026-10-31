# -*- coding: utf-8 -*-
"""Small exact tests used in the paper, without a scipy version dependency.

`wilcoxon_exact` computes the two-sided p-value of the signed-rank test under
the exact null distribution (all 2^n sign flips equiprobable). scipy only grew
`method="exact"` after 1.3, and versions differ, so the paper's p-values were
computed here to keep the replication independent of the scipy build.
"""
from __future__ import print_function

import numpy as np
from scipy import stats


def _rank_sum_distribution(n):
    """Distribution of the sum of a random subset of {1, ..., n}."""
    dist = {0: 1.0}
    for k in range(1, n + 1):
        nxt = {}
        for s, c in dist.items():
            nxt[s] = nxt.get(s, 0) + c
            nxt[s + k] = nxt.get(s + k, 0) + c
        dist = nxt
    total = float(2 ** n)
    return {s: c / total for s, c in dist.items()}


def wilcoxon_exact(x, y=None):
    """Two-sided exact Wilcoxon signed-rank test. Zeros are dropped."""
    x = np.asarray(x, dtype=float)
    if y is not None:
        x = x - np.asarray(y, dtype=float)
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return dict(p=1.0, statistic=0.0, n=0)
    if n > 20:
        r = stats.rankdata(np.abs(x))
        w, p = stats.wilcoxon(x)
        return dict(p=float(p), statistic=float(w), n=n, method="normal approximation")
    ranks = stats.rankdata(np.abs(x))
    w_plus = float(ranks[x > 0].sum())
    dist = _rank_sum_distribution(n)
    le = sum(c for s, c in dist.items() if s <= w_plus + 1e-9)
    ge = sum(c for s, c in dist.items() if s >= w_plus - 1e-9)
    return dict(p=min(1.0, 2 * min(le, ge)), statistic=w_plus, n=n, method="exact")
