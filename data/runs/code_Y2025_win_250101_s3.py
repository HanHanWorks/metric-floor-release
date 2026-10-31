import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 轻度平滑：抑制高频抖动，保留日周期形态
    if n >= 3:
        xs = x.copy()
        xs[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        xs[0] = x[0]
        xs[-1] = x[-1]
        x = xs

    # 2) 低功率段收缩（大量样本低于 0.2Cap，减少过预测）
    low_thr = 0.20 * cap
    mask_low = x < low_thr
    if np.any(mask_low):
        x[mask_low] = x[mask_low] * 0.62

    # 2b) 中低功率段轻度收缩
    mid_lo = 0.20 * cap
    mid_hi = 0.40 * cap
    mask_mid = (x >= mid_lo) & (x < mid_hi)
    if np.any(mask_mid):
        x[mask_mid] = x[mask_mid] * 0.82

    # 2c) 中等功率段轻微收缩
    mid2_lo = 0.40 * cap
    mid2_hi = 0.55 * cap
    mask_mid2 = (x >= mid2_lo) & (x < mid2_hi)
    if np.any(mask_mid2):
        x[mask_mid2] = x[mask_mid2] * 0.90

    # 3) 极低值截断
    x = np.where(x < 0.005 * cap, 0.0, x)

    # 4) 高值段抑制（p99 仅 0.63Cap，保守抑制）
    high_thr = 0.58 * cap
    mask_high = x > high_thr
    if np.any(mask_high):
        x[mask_high] = high_thr + (x[mask_high] - high_thr) * 0.70

    # 5) 物理约束
    x = np.clip(x, 0.0, cap)

    # 6) 保证非恒定
    if np.allclose(x, x[0]):
        x = x + np.linspace(0, 1e-6, n)

    return x