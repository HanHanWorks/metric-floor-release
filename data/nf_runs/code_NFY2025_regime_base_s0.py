import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    low = pp < 0.10 * cap
    mid = (pp >= 0.10 * cap) & (pp < 0.60 * cap)
    high = pp >= 0.60 * cap

    # 邻域均值（前后各1点）
    if n >= 3:
        nb = np.zeros(n)
        nb[1:-1] = 0.5 * (pp[:-2] + pp[2:])
        nb[0] = pp[1]
        nb[-1] = pp[-2]
    else:
        nb = pp.copy()

    # 低出力段：轻微收缩
    out = np.where(low, np.maximum(pp * 0.98, 0.0), out)

    # 中出力段：轻度平滑
    if n >= 3:
        smooth = pp.copy()
        smooth[1:-1] = 0.20 * pp[:-2] + 0.60 * pp[1:-1] + 0.20 * pp[2:]
        smooth[0] = 0.7 * pp[0] + 0.3 * pp[1]
        smooth[-1] = 0.7 * pp[-1] + 0.3 * pp[-2]
        out = np.where(mid, 0.6 * pp + 0.4 * smooth, out)

    # 高出力段：轻微衰减
    out = np.where(high, np.minimum(pp * 0.97, cap), out)

    # 孤立抖动修正
    if n >= 3:
        prev_diff = np.zeros(n)
        next_diff = np.zeros(n)
        prev_diff[1:] = pp[1:] - pp[:-1]
        next_diff[:-1] = pp[:-1] - pp[1:]

        threshold = 0.02 * cap
        isolated = np.zeros(n, dtype=bool)
        isolated[1:-1] = (prev_diff[1:-1] * next_diff[1:-1] > 0) & \
                         (np.abs(prev_diff[1:-1]) > threshold) & \
                         (np.abs(next_diff[1:-1]) > threshold)

        apply_mask = isolated & (~high)
        if np.any(apply_mask):
            out = np.where(apply_mask, 0.7 * out + 0.3 * nb, out)

    # 日尺度基线校正
    if n >= 96:
        win = 96
        valid = (~high).astype(float)
        x = pp * valid
        cs = np.cumsum(np.concatenate([[0.0], x]))
        cv = np.cumsum(np.concatenate([[0.0], valid]))
        base = np.zeros(n)
        half = win // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            s = cs[hi] - cs[lo]
            c = cv[hi] - cv[lo]
            base[i] = s / c if c > 0.5 else pp[i]

        dev = pp - base
        dev_thr = 0.15 * cap
        pull = mid & (np.abs(dev) > dev_thr)
        if np.any(pull):
            out = np.where(pull, out - 0.25 * dev, out)

    # 低出力段邻域一致性
    if n >= 3:
        close_to_neighbor = np.abs(pp - nb) < 0.05 * cap
        low_shrink = low & close_to_neighbor
        if np.any(low_shrink):
            out = np.where(low_shrink, 0.85 * out + 0.15 * nb, out)

    # 局部趋势一致性
    if n >= 5:
        lin_ref = np.zeros(n)
        lin_ref[2:-2] = 0.5 * (pp[:-4] + pp[4:])
        lin_ref[0] = pp[0]
        lin_ref[1] = pp[1]
        lin_ref[-2] = pp[-2]
        lin_ref[-1] = pp[-1]

        dev_lin = np.abs(pp - lin_ref)
        thr_lin = np.where(low, 0.04 * cap, 0.06 * cap)
        spike = (~high) & (dev_lin > thr_lin) & (dev_lin < 0.30 * cap)
        if np.any(spike):
            pull_amt = np.clip(0.30 * (pp - lin_ref), -0.03 * cap, 0.03 * cap)
            out = np.where(spike, out - pull_amt, out)

    # 极低出力段微调
    if n >= 3:
        tiny = pp < 0.03 * cap
        if np.any(tiny):
            nb_low = nb < 0.05 * cap
            tiny_shrink = tiny & nb_low
            if np.any(tiny_shrink):
                out = np.where(tiny_shrink, out * 0.90, out)

    # 中段平坦区轻微向上补偿
    mid_band = (pp >= 0.15 * cap) & (pp < 0.55 * cap)
    if np.any(mid_band) and n >= 3:
        flat = np.abs(pp - nb) < 0.02 * cap
        comp = mid_band & flat
        if np.any(comp):
            out = np.where(comp, out + 0.005 * cap, out)

    # 日周期峰值削顶保护
    if n >= 96:
        hour = view['hour'].values
        hour_bin = np.clip(np.floor(hour).astype(int), 0, 23)
        med = np.zeros(n)
        cnt = np.zeros(n)
        half = 96
        for i in range(n):
            h = hour_bin[i]
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            idx = np.arange(lo, hi)
            same_h = idx[hour_bin[idx] == h]
            same_h_mid = same_h[mid[same_h]]
            if len(same_h_mid) >= 3:
                med[i] = np.median(pp[same_h_mid])
                cnt[i] = len(same_h_mid)
            else:
                med[i] = pp[i]
        peak_excess = mid & (cnt >= 3) & (pp > med + 0.10 * cap)
        if np.any(peak_excess):
            out = np.where(peak_excess, out - 0.02 * cap, out)

    # 低出力段缓慢漂移抑制
    if n >= 9:
        long_nb = np.zeros(n)
        cs = np.cumsum(np.concatenate([[0.0], pp]))
        for i in range(n):
            lo = max(0, i - 4)
            hi = min(n, i + 5)
            s = cs[hi] - cs[lo]
            c = hi - lo
            if c > 1:
                long_nb[i] = (s - pp[i]) / (c - 1)
            else:
                long_nb[i] = pp[i]
        drift = low & (np.abs(pp - long_nb) > 0.03 * cap) & (np.abs(pp - long_nb) < 0.10 * cap)
        if np.any(drift):
            out = np.where(drift, 0.80 * out + 0.20 * long_nb, out)

    # 中高出力段平台期轻微下修
    upper_mid = (pp >= 0.40 * cap) & (pp < 0.60 * cap)
    if np.any(upper_mid) and n >= 3:
        local_bump = (pp > nb) & ((pp - nb) < 0.08 * cap)
        bump_mask = upper_mid & local_bump
        if np.any(bump_mask):
            out = np.where(bump_mask, out - 0.008 * cap, out)

    # 极低出力段轻微上调
    very_low = pp < 0.05 * cap
    if np.any(very_low):
        lift = very_low & (pp > 0.005 * cap)
        if np.any(lift):
            out = np.where(lift, out + 0.002 * cap, out)

    # 中段稳定平台区轻微上调
    plateau_band = (pp >= 0.20 * cap) & (pp < 0.45 * cap)
    if np.any(plateau_band) and n >= 5:
        stable = np.zeros(n, dtype=bool)
        ref4 = np.zeros(n)
        ref4[2:-2] = 0.5 * (pp[:-4] + pp[4:])
        ref4[0] = pp[0]
        ref4[1] = pp[1]
        ref4[-2] = pp[-2]
        ref4[-1] = pp[-1]
        stable[2:-2] = np.abs(pp[2:-2] - ref4[2:-2]) < 0.015 * cap
        plat_mask = plateau_band & stable
        if np.any(plat_mask):
            out = np.where(plat_mask, out + 0.004 * cap, out)

    # 中高出力段拐点保护
    knee_band = (pp >= 0.50 * cap) & (pp < 0.62 * cap)
    if np.any(knee_band) and n >= 3:
        rising_edge = (pp > nb) & ((pp - nb) < 0.03 * cap)
        knee_mask = knee_band & rising_edge
        if np.any(knee_mask):
            out = np.where(knee_mask, out - 0.004 * cap, out)

    # 中段（0.10~0.20cap）缓慢爬坡区轻微上调
    ramp_band = (pp >= 0.10 * cap) & (pp < 0.20 * cap)
    if np.any(ramp_band) and n >= 5:
        rising = np.zeros(n, dtype=bool)
        if n >= 3:
            rising[2:] = (pp[2:] >= pp[1:-1]) & (pp[1:-1] >= pp[:-2])
        ramp_mask = ramp_band & rising
        if np.any(ramp_mask):
            out = np.where(ramp_mask, out + 0.003 * cap, out)

    # 中高出力段（0.30~0.50cap）稳定平台轻微上修
    plat2_band = (pp >= 0.30 * cap) & (pp < 0.50 * cap)
    if np.any(plat2_band) and n >= 5:
        stable2 = np.zeros(n, dtype=bool)
        ref2 = np.zeros(n)
        ref2[2:-2] = 0.5 * (pp[:-4] + pp[4:])
        ref2[0] = pp[0]
        ref2[1] = pp[1]
        ref2[-2] = pp[-2]
        ref2[-1] = pp[-1]
        stable2[2:-2] = np.abs(pp[2:-2] - ref2[2:-2]) < 0.012 * cap
        plat2_mask = plat2_band & stable2
        if np.any(plat2_mask):
            out = np.where(plat2_mask, out + 0.003 * cap, out)

    # 低出力段（0.05~0.10cap）稳定平台轻微上调
    low_band = (pp >= 0.05 * cap) & (pp < 0.10 * cap)
    if np.any(low_band) and n >= 5:
        stable_low = np.zeros(n, dtype=bool)
        ref_low = np.zeros(n)
        ref_low[2:-2] = 0.5 * (pp[:-4] + pp[4:])
        ref_low[0] = pp[0]
        ref_low[1] = pp[1]
        ref_low[-2] = pp[-2]
        ref_low[-1] = pp[-1]
        stable_low[2:-2] = np.abs(pp[2:-2] - ref_low[2:-2]) < 0.010 * cap
        low_plat = low_band & stable_low
        if np.any(low_plat):
            out = np.where(low_plat, out + 0.002 * cap, out)

    # 中段（0.20~0.40cap）缓慢下降趋势轻微下调
    down_band = (pp >= 0.20 * cap) & (pp < 0.40 * cap)
    if np.any(down_band) and n >= 5:
        falling = np.zeros(n, dtype=bool)
        falling[2:] = (pp[2:] <= pp[1:-1]) & (pp[1:-1] <= pp[:-2])
        small_drop = np.zeros(n, dtype=bool)
        small_drop[2:] = (pp[1:-1] - pp[2:]) < 0.02 * cap
        down_mask = down_band & falling & small_drop
        if np.any(down_mask):
            out = np.where(down_mask, out - 0.002 * cap, out)

    # 新增：日周期谷段（凌晨/夜间低出力时段）整体轻微上修
    # 日前预测在夜间谷段常系统偏低，稳定非零谷底上调有助ACC。
    if n >= 96:
        hour = view['hour'].values
        # 夜间低出力时段（22-06 点）
        night = (hour >= 22.0) | (hour < 6.0)
        night_low = night & (pp >= 0.03 * cap) & (pp < 0.15 * cap)
        if np.any(night_low):
            # 仅当该点并非局部凸起（避免引入抖动）
            smooth_ok = np.abs(pp - nb) < 0.03 * cap
            night_mask = night_low & smooth_ok
            if np.any(night_mask):
                out = np.where(night_mask, out + 0.0025 * cap, out)

    # 新增：日周期午间高发时段（10-14点）中低出力轻微上修
    # 该时段日前预测在中低出力下常偏保守。
    if n >= 96:
        hour = view['hour'].values
        noon = (hour >= 10.0) & (hour < 14.0)
        noon_mid = noon & (pp >= 0.12 * cap) & (pp < 0.35 * cap)
        if np.any(noon_mid):
            stable_n = np.abs(pp - nb) < 0.02 * cap
            noon_mask = noon_mid & stable_n
            if np.any(noon_mask):
                out = np.where(noon_mask, out + 0.002 * cap, out)

    # 新增：傍晚快速下降沿（16-19点）轻微下修
    # 日前预测在傍晚爬坡下降过程常滞后，轻微下修贴近真实。
    if n >= 5:
        hour = view['hour'].values
        eve = (hour >= 16.0) & (hour < 19.0)
        eve_mid = eve & mid
        if np.any(eve_mid):
            falling = np.zeros(n, dtype=bool)
            falling[1:] = pp[1:] < pp[:-1]
            eve_mask = eve_mid & falling & ((pp - nb) > -0.02 * cap)
            if np.any(eve_mask):
                out = np.where(eve_mask, out - 0.002 * cap, out)

    # 全局裁剪
    out = np.clip(out, 0.0, cap)

    if np.allclose(out, out[0]):
        out = pp.copy()

    return out