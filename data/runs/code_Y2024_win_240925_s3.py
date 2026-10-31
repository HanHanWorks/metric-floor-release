import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 极轻量平滑：step_autocorr 极高，序列本身已很平滑，仅做非常温和的 3 点加权。
    if n >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        padded = np.pad(pp, 1, mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        out = 0.75 * out + 0.25 * smooth

    # 2) 日周期一致性修正：向同 hour 跨天中位数温和收缩。
    #    但 PP 分布高度偏斜（p50=0.10cap, p90=0.29cap），中位数收缩会系统性
    #    拉低峰值。改为仅在偏离中位数较大时才收缩，且收缩幅度随偏离增大而增大，
    #    对接近中位数的点几乎不动，保留峰值信息。
    if n >= 96 * 3:
        day_len = 96
        n_days = n // day_len
        reshaped = out[:n_days * day_len].reshape(n_days, day_len)
        hour_median = np.median(reshaped, axis=0)
        dev = reshaped - hour_median[None, :]
        # 自适应收缩：偏离越大收缩越强，最大收缩 20%
        scale = np.maximum(np.abs(hour_median[None, :]), 1e-3)
        rel_dev = np.abs(dev) / scale
        shrink = 0.20 * np.clip(rel_dev, 0.0, 1.0)
        adjusted = reshaped - shrink * dev
        out[:n_days * day_len] = adjusted.reshape(-1)

    # 3) 物理约束：非负，且不超过装机容量
    hi = cap if np.isfinite(cap) and cap > 0 else None
    out = np.clip(out, 0.0, hi)

    # 4) 避免常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out