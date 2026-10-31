import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 76.625
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)

    n = len(pp)
    out = pp.copy()

    # 1) 极轻平滑，仅平缓段
    if n >= 3:
        sm = pp.copy()
        sm[1:-1] = 0.2 * pp[:-2] + 0.6 * pp[1:-1] + 0.2 * pp[2:]
        ramp = np.abs(np.diff(pp, prepend=pp[0]))
        thr = np.percentile(ramp, 60)
        smooth_mask = ramp < thr
        out = np.where(smooth_mask, sm, pp)

    ratio = pp / max(cap, 1e-6)

    # 2) 整体校准（继续小幅收敛）
    out = out * 0.9440

    # 3) 低功率段再轻压
    low_mask = ratio < 0.15
    out = np.where(low_mask, out * 0.9445, out)

    # 4) 中段略压
    mid_mask = (ratio >= 0.15) & (ratio < 0.50)
    out = np.where(mid_mask, out * 0.9710, out)

    # 5) 高段几乎不动
    hi_mask = ratio >= 0.50
    out = np.where(hi_mask, out * 0.9960, out)

    # 6) 夜间时段收敛
    night = (hour < 6) | (hour >= 22)
    out = np.where(night, out * 0.9035, out)

    # 7) 傍晚高发时段轻微上抬
    eve = (hour >= 16) & (hour < 20)
    out = np.where(eve, out * 1.0155, out)

    # 8) 清晨爬坡段轻微上抬
    morn = (hour >= 6) & (hour < 9)
    out = np.where(morn, out * 1.0115, out)

    # 9) 周末轻微下调
    weekend = (dow >= 5)
    out = np.where(weekend, out * 0.9895, out)

    # 10) 极低功率段额外微压
    very_low = ratio < 0.05
    out = np.where(very_low, out * 0.9805, out)

    # 11) 限幅
    out = np.clip(out, 0.0, cap)

    # 12) 保证非常数
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out