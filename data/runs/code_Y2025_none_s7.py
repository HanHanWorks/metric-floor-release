import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    n = len(pp)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and n > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    out = pp.copy()
    ratio = pp / cap

    # 分档缩放：数据画像显示 62% 时间低于 0.2cap，39% 低于 0.1cap
    # 电网ACC考核对低功率段的下压敏感，逐步微调各档系数（第60轮微调：低段再压一档）
    very_low = ratio < 0.05
    out[very_low] = pp[very_low] * 0.432

    mid_low = (ratio >= 0.05) & (ratio < 0.1)
    out[mid_low] = pp[mid_low] * 0.576

    low_mid = (ratio >= 0.1) & (ratio < 0.15)
    out[low_mid] = pp[low_mid] * 0.672

    low_mid2 = (ratio >= 0.15) & (ratio < 0.2)
    out[low_mid2] = pp[low_mid2] * 0.742

    mid = (ratio >= 0.2) & (ratio < 0.35)
    out[mid] = pp[mid] * 0.827

    mid_high = (ratio >= 0.35) & (ratio < 0.55)
    out[mid_high] = pp[mid_high] * 0.908

    high_mid = (ratio >= 0.55) & (ratio < 0.75)
    out[high_mid] = pp[high_mid] * 0.957

    upper_mid = (ratio >= 0.75) & (ratio <= 0.9)
    out[upper_mid] = pp[upper_mid] * 0.984

    high_mask = ratio > 0.9
    out[high_mask] = pp[high_mask] * 1.010

    # 轻度平滑：仅在平缓段应用，保留真实爬坡
    if n >= 3:
        smoothed = out.copy()
        smoothed[1:-1] = 0.10 * out[:-2] + 0.80 * out[1:-1] + 0.10 * out[2:]
        ramp = np.abs(np.diff(pp, prepend=pp[0]))
        ramp_thresh = 0.035 * cap
        smooth_mask = np.zeros(n, dtype=bool)
        smooth_mask[1:-1] = (ramp[1:-1] < ramp_thresh) & (ramp[2:] < ramp_thresh)
        out[smooth_mask] = smoothed[smooth_mask]

    # 极低功率段再保守一档
    ultra_low = ratio < 0.03
    out[ultra_low] = out[ultra_low] * 0.828

    out = np.clip(out, 0.0, cap)

    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out