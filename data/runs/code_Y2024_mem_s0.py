import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    n = len(pp)
    if n == 0:
        return pp

    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    def smooth(x, w):
        if w <= 1 or len(x) < 3:
            return x
        w = int(w)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        xp = np.pad(x, pad, mode='reflect')
        k = np.ones(w) / w
        return np.convolve(xp, k, mode='valid')

    # 更强平滑：step_autocorr 极高，进一步压噪有利于 ACC
    sm = smooth(pp, 7)

    ratio = sm / cap

    # 低出力段收缩（保守，主体在低段）
    shrink = np.ones_like(ratio)
    shrink = np.where(ratio < 0.02, 0.62, shrink)
    shrink = np.where((ratio >= 0.02) & (ratio < 0.05), 0.72, shrink)
    shrink = np.where((ratio >= 0.05) & (ratio < 0.10), 0.82, shrink)
    shrink = np.where((ratio >= 0.10) & (ratio < 0.18), 0.90, shrink)
    shrink = np.where((ratio >= 0.18) & (ratio < 0.28), 0.95, shrink)
    shrink = np.where(ratio >= 0.28, 0.985, shrink)

    sm = sm * shrink
    sm = np.clip(sm, 0.0, None)

    # 以平滑为主，保留少量原始结构
    alpha = 0.90
    out = alpha * sm + (1.0 - alpha) * pp

    # 二次轻平滑
    out = smooth(out, 5)

    # 整体校准（略为下调，因历史显示过预测略有偏高风险）
    out = out * 0.978

    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
    if hour is not None and len(hour) == n:
        night = ((hour < 6.0) | (hour >= 19.0))
        midday = ((hour >= 9.0) & (hour <= 16.0))
        adj = np.ones_like(out)
        adj = np.where(night, 0.965, adj)
        adj = np.where(midday, 1.014, adj)
        out = out * adj

    # 温和爬坡抑制
    ramp_p99 = 0.0296
    max_step = ramp_p99 * cap * 1.5
    if n > 1:
        out = out.copy()
        for i in range(1, n):
            d = out[i] - out[i - 1]
            if d > max_step:
                out[i] = out[i - 1] + max_step
            elif d < -max_step:
                out[i] = out[i - 1] - max_step
        out = smooth(out, 3)

    out = np.clip(out, 0.0, cap)

    if np.std(out) < 1e-9 and np.std(pp) > 1e-9:
        out = pp.copy()

    return out