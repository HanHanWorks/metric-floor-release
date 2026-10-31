import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 保守去噪：仅对噪声级的小幅波动做三点平滑，保留真实爬坡
    if n >= 3:
        sm = x.copy()
        for i in range(1, n - 1):
            a, b, c = x[i - 1], x[i], x[i + 1]
            if (b - a) * (c - b) >= 0:
                sm[i] = 0.25 * a + 0.5 * b + 0.25 * c
            else:
                sm[i] = b
        diff = np.abs(sm - x)
        mask = diff < 0.008 * cap
        x = np.where(mask, sm, x)

    # 2) 物理裁剪
    x = np.clip(x, 0.0, cap)

    # 3) 低功率段温和收缩（数据画像：61% < 0.2cap, 33% < 0.1cap）
    #    分档收缩，越低越收缩，避免夜间虚高
    low1 = x < 0.10 * cap
    low2 = (x >= 0.10 * cap) & (x < 0.20 * cap)
    x = np.where(low1, x * 0.95, x)
    x = np.where(low2, x * 0.975, x)

    # 4) 高功率段轻微抑制（>0.9cap 仅 1.8%）
    high = x > 0.90 * cap
    x = np.where(high, 0.90 * cap + (x - 0.90 * cap) * 0.92, x)

    # 5) 防止常数序列
    if np.allclose(x, x[0]):
        x = pp.copy()
        x = np.clip(x, 0.0, cap)
        # 若原始也近似常数，加入极小扰动
        if np.allclose(x, x[0]):
            eps = 1e-6 * cap
            x = x + eps * np.sin(np.arange(n) * 0.1)

    return x