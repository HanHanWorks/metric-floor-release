import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑：抑制高频抖动
    if n >= 5:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel = kernel / kernel.sum()
        pad = 2
        padded = np.pad(pp, (pad, pad), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        ratio = pp / cap
        w = np.clip(1.0 - ratio / 0.3, 0.0, 1.0)
        out = w * smooth + (1.0 - w) * pp

    # 2) 物理约束
    out = np.clip(out, 0.0, cap)

    # 3) 夜间归零修正
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night = ((hour < 6.0) | (hour >= 22.0)).astype(float)
        low = np.clip(1.0 - (out / cap) / 0.15, 0.0, 1.0)
        shrink = night * low * 0.5
        out = out * (1.0 - shrink)

    # 4) 低功率段轻微向零收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        low_global = np.clip(1.0 - (out / cap) / 0.10, 0.0, 1.0)
        day_shrink = low_global * 0.15
        out = out * (1.0 - day_shrink)

    # 5) 平滑序列级偏差收缩（利用高 step_autocorr，逐点向局部均值靠拢）
    if n >= 3:
        # 三点局部均值，仅在低功率段应用，减弱噪声引起的尖峰
        padded = np.pad(out, (1, 1), mode='edge')
        local_mean = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        low_mask = np.clip(1.0 - (out / cap) / 0.20, 0.0, 1.0)
        beta = 0.10 * low_mask
        out = out * (1.0 - beta) + local_mean * beta

    # 6) 防止常数输出
    if np.std(out) < 1e-6:
        t = np.arange(n, dtype=float)
        out = out + 0.001 * cap * np.sin(2.0 * np.pi * t / max(n, 1))

    out = np.clip(out, 0.0, cap)
    return out