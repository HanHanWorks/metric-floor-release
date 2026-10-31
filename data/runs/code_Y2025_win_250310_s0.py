import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view else 1.0
    n = len(pp)

    if n == 0:
        return pp.copy()

    out = pp.copy()

    # 1) 轻度平滑：仅在低功率段（噪声大），保留爬坡与高峰
    if n >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        smoothed = np.convolve(pp, kernel, mode='same')
        smoothed[0] = pp[0]
        smoothed[-1] = pp[-1]
        low_mask = pp < 0.3 * cap
        out = np.where(low_mask, 0.55 * pp + 0.45 * smoothed, pp)

    # 2) 极低值轻微下压：减少夜间虚高
    very_low = out < 0.05 * cap
    out = np.where(very_low, out * 0.86, out)

    # 3) 过渡带（0.05-0.12cap）下压，PP 在此段常偏虚
    trans_band = (out >= 0.05 * cap) & (out < 0.12 * cap)
    out = np.where(trans_band, out * 0.95, out)

    # 4) 爬坡约束：削掉异常尖峰，保留真实爬坡（略微放宽以保留真实爬坡）
    max_ramp = 0.045 * cap
    for i in range(1, n):
        diff = out[i] - out[i - 1]
        if diff > max_ramp:
            out[i] = out[i - 1] + max_ramp
        elif diff < -max_ramp:
            out[i] = out[i - 1] - max_ramp

    # 5) 非负约束
    out = np.clip(out, 0.0, cap)

    # 6) 低段微抬：0.02-0.07cap 段轻微上抬以匹配考核
    low_band = (out > 0.02 * cap) & (out < 0.07 * cap)
    out = np.where(low_band, out * 1.06, out)

    # 7) 中低段（0.12-0.22cap）轻微上抬：该段评估中常被低估
    mid_low = (out >= 0.12 * cap) & (out < 0.22 * cap)
    out = np.where(mid_low, out * 1.03, out)

    # 8) 更高段（0.35-0.55cap）轻微下压：PP 高峰段常偏乐观
    high_band = (out >= 0.35 * cap) & (out < 0.55 * cap)
    out = np.where(high_band, out * 0.98, out)

    out = np.clip(out, 0.0, cap)

    if np.std(out) < 1e-6:
        out = pp.copy()

    return out