import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 极轻平滑：5点加权，权重更集中在中心，尽量保留爬坡
    if n >= 5:
        xs = x.copy()
        xs[2:-2] = (0.1 * x[:-4] + 0.2 * x[1:-3] + 0.4 * x[2:-2]
                    + 0.2 * x[3:-1] + 0.1 * x[4:])
        x = xs
    elif n >= 3:
        xs = x.copy()
        xs[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        x = xs

    # 2) 尖峰温和收缩：仅对超过局部中位数较大偏差的点做收缩
    if n >= 7:
        pad = np.pad(x, 3, mode='edge')
        med = np.empty(n)
        for i in range(n):
            med[i] = np.median(pad[i:i+7])
        dev = x - med
        mad = np.median(np.abs(dev)) + 1e-6
        thresh = 5.0 * mad
        shrink = np.clip(dev, -thresh, thresh)
        x = med + shrink

    # 3) 容量约束
    if np.isfinite(cap) and cap > 0:
        x = np.clip(x, 0.0, cap)

    # 4) 负值归零
    x = np.where(x < 0, 0.0, x)

    # 5) 保守回归原预测（高 alpha，减少改动幅度）
    alpha = 0.93
    out = alpha * x + (1 - alpha) * pp

    # 6) 再次容量约束
    if np.isfinite(cap) and cap > 0:
        out = np.clip(out, 0.0, cap)

    # 7) 防止常数序列
    if np.allclose(out, out[0]):
        out = pp.copy()

    return out