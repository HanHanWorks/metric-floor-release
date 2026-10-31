import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view else 100.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 中值滤波去孤立尖峰（M03）
    if n >= 3:
        med = np.median(np.stack([pp, np.roll(pp, 1), np.roll(pp, -1)]), axis=0)
        med[0] = np.median([pp[0], pp[1]]) if n > 1 else pp[0]
        med[-1] = np.median([pp[-1], pp[-2]]) if n > 1 else pp[-1]
        dev = np.abs(pp - med)
        thr = max(0.05 * cap, 3.0 * np.median(dev) + 1e-6)
        mask = dev > thr
        out[mask] = med[mask]

    # 2) 轻度加权三点平滑（M02）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 3) 容量约束（M07）
    out = np.clip(out, 0.0, cap)

    # 4) 爬坡限制（M08），阈值贴近 ramp_p99
    if n >= 2:
        max_ramp = 0.06 * cap
        for i in range(1, n):
            d = out[i] - out[i - 1]
            if d > max_ramp:
                out[i] = out[i - 1] + max_ramp
            elif d < -max_ramp:
                out[i] = out[i - 1] - max_ramp
        out = np.clip(out, 0.0, cap)

    # 5) 低出力区保守上抬：仅在明显低于当日中位且接近零时，轻微抬升，
    #    避免过度零偏同时不破坏原有低出力分布
    if n >= 4:
        day_med = np.median(out)
        low_mask = (out < 0.02 * cap) & (day_med > 0.05 * cap)
        if np.any(low_mask):
            lift = 0.005 * cap
            out[low_mask] = np.minimum(out[low_mask] + lift, 0.02 * cap)

    # 6) 保底：避免常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out