import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度中心加权平滑，抑制毛刺、保留日内形态
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 2) 尖峰回拉：只处理连续大跳变的孤立尖峰
    if n >= 3:
        diff = np.diff(out)
        ramp_thr = 0.05 * cap
        for i in range(1, n - 1):
            if abs(diff[i - 1]) > ramp_thr and abs(diff[i]) > ramp_thr:
                if np.sign(diff[i - 1]) != np.sign(diff[i]):
                    out[i] = 0.5 * (out[i - 1] + out[i + 1])

    # 3) 低出力段轻微收缩，减少夜间小值噪声
    low_mask = out < 0.10 * cap
    out[low_mask] = out[low_mask] * 0.97

    very_low_mask = out < 0.03 * cap
    out[very_low_mask] = out[very_low_mask] * 0.95

    # 4) clip 到 [0, cap]
    out = np.clip(out, 0.0, cap)

    # 5) 保证非恒定
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out