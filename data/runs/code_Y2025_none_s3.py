import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    x = pp / cap
    n = len(x)
    if n == 0:
        return pp

    # 1) 极低值软阈值：抑制噪声性小波动
    low = 0.020
    x = np.where(x < low, x * (x / low), x)

    # 2) 轻度平滑（step_autocorr 极高，平滑安全）
    if n >= 3:
        k = np.array([0.2, 0.6, 0.2])
        pad = np.concatenate(([x[0]], x, [x[-1]]))
        xs = np.convolve(pad, k, mode='valid')
        x = xs

    # 3) 轻度收缩到均值，降低方差
    mu = np.mean(x)
    x = 0.93 * x + 0.07 * mu

    # 4) 保守下调，ACC 对高估更敏感
    x = 0.925 * x

    # 5) 轻微削峰：对高分段做柔性压缩，避免大幅高估拖累 ACC
    hi = 0.48
    x = np.where(x > hi, hi + (x - hi) * 0.85, x)

    # 6) 物理范围裁剪
    x = np.clip(x, 0.0, 1.0)

    out = x * cap

    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out