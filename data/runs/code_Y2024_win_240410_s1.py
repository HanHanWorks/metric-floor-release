import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    out = pp.copy()

    if n == 0:
        return out

    # 1) 轻度平滑（保留原值主导，抑制高频抖动）
    if n >= 5:
        kernel = np.array([0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05])
        # 归一化
        kernel = kernel / kernel.sum()
        pad = len(kernel) // 2
        padded = np.pad(out, (pad, pad), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        if len(smoothed) != n:
            smoothed = smoothed[:n]
        out = 0.7 * out + 0.3 * smoothed

    # 2) 物理约束
    out = np.clip(out, 0.0, cap)

    # 3) 极小值抑制（低于 0.5% cap 视为噪声归零）
    eps = 0.005 * cap
    out = np.where(out < eps, 0.0, out)

    # 4) 保证非恒定
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out