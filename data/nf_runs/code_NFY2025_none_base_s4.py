import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻量平滑：抑制高频抖动（step_autocorr 极高，序列本身平滑）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.15 * out[:-2] + 0.70 * out[1:-1] + 0.15 * out[2:]
        out = sm

    # 2) 低出力段收缩：大量样本低于 0.2cap，低段预测噪声大，向 0 轻微收缩
    ratio = out / max(cap, 1e-6)
    low_mask = ratio < 0.20
    out[low_mask] = out[low_mask] * 0.90

    # 3) 极低出力软截断：避免近零噪声/负值
    tiny = out < 0.01 * cap
    out[tiny] = np.maximum(out[tiny], 0.0)

    # 4) 爬坡限幅：ramp_p99 约 0.048cap，限制单步变化
    max_ramp = 0.06 * cap
    if n >= 2:
        for i in range(1, n):
            d = out[i] - out[i-1]
            if d > max_ramp:
                out[i] = out[i-1] + max_ramp
            elif d < -max_ramp:
                out[i] = out[i-1] - max_ramp

    # 5) 物理边界
    out = np.clip(out, 0.0, cap)

    # 6) 保底：若退化为常数，回退原始预测
    if np.nanstd(out) < 1e-9 and np.nanstd(pp) > 1e-9:
        out = pp.copy()

    return out