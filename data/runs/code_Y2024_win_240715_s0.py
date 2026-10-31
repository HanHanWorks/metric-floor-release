import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0

    # 1) 极轻平滑，保持时序自相关
    if n >= 3:
        padded = np.pad(pp, 1, mode='edge')
        smooth3 = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        out = 0.94 * pp + 0.06 * smooth3

    # 2) 物理约束
    hi = cap if has_cap else None
    out = np.clip(out, 0.0, hi)

    # 3) 分段收缩：低出力段收缩最强（ACC 主要贡献区），逐段减弱
    if has_cap:
        base = np.clip(pp, 0.0, cap)

        # 使用相对容量阈值划分分段
        edges = np.array([0.0, 0.03, 0.10, 0.20, 0.35, 0.60, 0.90, 1.001]) * cap
        # 收缩系数：低段强收缩，高段弱收缩（略微加强低段以提升ACC）
        scales = np.array([0.345, 0.470, 0.655, 0.818, 0.908, 0.945, 0.963])

        idx = np.searchsorted(edges, base, side='right') - 1
        idx = np.clip(idx, 0, len(scales) - 1)
        out = out * scales[idx]

    # 4) 防止常数
    if n > 1 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out