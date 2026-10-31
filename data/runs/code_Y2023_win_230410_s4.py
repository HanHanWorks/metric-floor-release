import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 83.875
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 极低值抬升：数据中约 48% 低于 0.1cap、1.7% 为 0，
    #    小幅抬升近零段以降低零/近零带来的考核惩罚，但避免过度抬升
    low_thr = 0.015 * cap
    lift = 0.006 * cap
    low_mask = out < low_thr
    out[low_mask] = np.maximum(out[low_mask], lift)

    # 2) 轻度平滑：ramp_p99 很小 (~0.023cap)，说明序列本身平缓，
    #    对平缓段做 3 点加权平滑可降噪，对陡变段保持原样
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        diff = np.abs(out[1:-1] - sm[1:-1])
        thr = 0.018 * cap
        mask = diff < thr
        out[1:-1] = np.where(mask, sm[1:-1], out[1:-1])

    # 3) 日内均值收缩：day_autocorr 仅 0.44，日内结构偏弱，
    #    适度向日内均值收缩以降方差，同时保留趋势
    pts_per_day = 96
    n_days = n // pts_per_day
    if n_days >= 1:
        for d in range(n_days):
            s = d * pts_per_day
            e = s + pts_per_day
            seg = out[s:e]
            seg_mean = seg.mean()
            out[s:e] = 0.90 * seg + 0.10 * seg_mean

    # 4) 全局均值轻微收缩，进一步稳定
    global_mean = out.mean()
    out = 0.96 * out + 0.04 * global_mean

    # 5) 裁剪到 [0, cap]
    out = np.clip(out, 0.0, cap)

    return out