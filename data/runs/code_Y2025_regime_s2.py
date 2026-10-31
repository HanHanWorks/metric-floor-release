import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view else 100.0
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)

    out = pp.copy()
    n = len(pp)
    if n == 0:
        return out

    low = pp < 0.20 * cap
    mid = (pp >= 0.20 * cap) & (pp < 0.60 * cap)
    high = pp >= 0.60 * cap

    ramp = np.zeros(n)
    if n > 1:
        ramp[1:] = np.abs(np.diff(pp))
    strong_ramp = ramp > 0.10 * cap

    # 局部中值（3点）
    med = pp.copy()
    if n >= 3:
        med[1:-1] = np.median(np.stack([pp[:-2], pp[1:-1], pp[2:]], axis=0), axis=0)
    elif n == 2:
        med[:] = np.median(pp)

    # 加权三点平滑
    smooth = pp.copy()
    if n >= 3:
        smooth[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]

    # 5点平滑
    smooth5 = pp.copy()
    if n >= 5:
        smooth5[2:-2] = (pp[:-4] + 2*pp[1:-3] + 3*pp[2:-2] + 2*pp[3:-1] + pp[4:]) / 9.0

    # ---------- 低出力段 (<20%Cap) 占59%：主战场 ----------
    if low.any():
        low_stable = low & (~strong_ramp)
        if low_stable.any():
            out[low_stable] = 0.50 * pp[low_stable] + 0.30 * med[low_stable] + 0.20 * smooth5[low_stable]
            out[low_stable] = out[low_stable] * 0.906

            very_low = low_stable & (pp < 0.10 * cap)
            if very_low.any():
                out[very_low] = out[very_low] * 0.916
            ultra_low = low_stable & (pp < 0.05 * cap)
            if ultra_low.any():
                out[ultra_low] = out[ultra_low] * 0.926
            tiny = low_stable & (pp < 0.03 * cap)
            if tiny.any():
                out[tiny] = out[tiny] * 0.936

            if low_stable.sum() >= 8:
                glob_mean = float(np.mean(pp[low_stable]))
                for h in np.unique(hour):
                    m = low_stable & (hour == h)
                    if m.sum() >= 4:
                        hm = float(np.mean(pp[m]))
                        if hm > glob_mean * 1.10:
                            out[m] = out[m] * 0.968
                        elif hm < glob_mean * 0.85:
                            out[m] = out[m] * 1.018

            if low_stable.sum() >= 8:
                glob_mean2 = float(np.mean(pp[low_stable]))
                for d in np.unique(dow):
                    m = low_stable & (dow == d)
                    if m.sum() >= 6:
                        dm = float(np.mean(pp[m]))
                        if dm > glob_mean2 * 1.08:
                            out[m] = out[m] * 0.977
                        elif dm < glob_mean2 * 0.88:
                            out[m] = out[m] * 1.013

            if n >= 3:
                local_dev = np.abs(pp - smooth5)
                unstable_low = low_stable & (local_dev > 0.03 * cap)
                if unstable_low.any():
                    out[unstable_low] = 0.55 * out[unstable_low] + 0.45 * med[unstable_low]

        low_ramp = low & strong_ramp
        if low_ramp.any():
            out[low_ramp] = 0.82 * pp[low_ramp] + 0.18 * med[low_ramp]

    # ---------- 中出力段 (20-60%Cap) 占32% ----------
    if mid.any():
        mid_stable = mid & (~strong_ramp)
        if mid_stable.any():
            out[mid_stable] = 0.55 * pp[mid_stable] + 0.25 * smooth[mid_stable] + 0.20 * med[mid_stable]
            out[mid_stable] = out[mid_stable] * 0.970

            thr = 0.02 * cap
            lift = mid & (out < thr)
            if lift.any():
                out[lift] = thr

            if mid_stable.sum() >= 8:
                glob_mid = float(np.mean(pp[mid_stable]))
                for h in np.unique(hour):
                    m = mid_stable & (hour == h)
                    if m.sum() >= 4:
                        hm = float(np.mean(pp[m]))
                        if hm > glob_mid * 1.08:
                            out[m] = out[m] * 0.985
                        elif hm < glob_mid * 0.85:
                            out[m] = out[m] * 1.012

        mid_ramp = mid & strong_ramp
        if mid_ramp.any():
            out[mid_ramp] = 0.90 * pp[mid_ramp] + 0.10 * smooth[mid_ramp]

    # ---------- 高出力段 (>60%Cap) ----------
    if high.any():
        high_stable = high & (~strong_ramp)
        if high_stable.any():
            out[high_stable] = 0.88 * pp[high_stable] + 0.12 * smooth[high_stable]
            out[high_stable] = out[high_stable] * 1.006
        high_ramp = high & strong_ramp
        if high_ramp.any():
            out[high_ramp] = 0.94 * pp[high_ramp] + 0.06 * med[high_ramp]

    # ---------- 强爬坡段兜底 ----------
    other_ramp = strong_ramp & (~low) & (~mid) & (~high)
    if other_ramp.any():
        out[other_ramp] = 0.92 * pp[other_ramp] + 0.08 * med[other_ramp]

    # ---------- 低段微调：稳定点再压 ----------
    if n >= 3 and low.any():
        low_stable = low & (~strong_ramp)
        if low_stable.any():
            dprev = np.zeros(n)
            dnext = np.zeros(n)
            dprev[1:] = np.abs(pp[1:] - pp[:-1])
            dnext[:-1] = np.abs(pp[:-1] - pp[1:])
            local_diff = np.maximum(dprev, dnext)
            very_stable = low_stable & (local_diff < 0.02 * cap)
            if very_stable.any():
                out[very_stable] = out[very_stable] * 0.976

            low_upper = low_stable & (pp >= 0.12 * cap) & (pp < 0.20 * cap)
            if low_upper.any():
                out[low_upper] = out[low_upper] * 0.980

            very_stable_ultra = very_stable & (pp < 0.06 * cap)
            if very_stable_ultra.any():
                out[very_stable_ultra] = out[very_stable_ultra] * 0.963

            low_edge = low_stable & (pp >= 0.17 * cap) & (pp < 0.20 * cap)
            if low_edge.any():
                out[low_edge] = out[low_edge] * 1.012

    # ---------- 极低段进一步收缩 ----------
    if low.any():
        low_stable2 = low & (~strong_ramp)
        if low_stable2.any():
            vlow_stable = low_stable2 & (pp < 0.08 * cap)
            if vlow_stable.any():
                out[vlow_stable] = out[vlow_stable] * 0.985

            low_top = low_stable2 & (pp >= 0.15 * cap) & (pp < 0.20 * cap)
            if low_top.any():
                out[low_top] = out[low_top] * 1.006

    # ---------- 中段稳定点轻微再压 ----------
    if mid.any():
        mid_stable2 = mid & (~strong_ramp)
        if mid_stable2.any():
            mid_center = mid_stable2 & (pp >= 0.30 * cap) & (pp < 0.45 * cap)
            if mid_center.any():
                out[mid_center] = out[mid_center] * 0.994

            mid_low_edge = mid_stable2 & (pp >= 0.20 * cap) & (pp < 0.25 * cap)
            if mid_low_edge.any():
                out[mid_low_edge] = out[mid_low_edge] * 0.992

    # ---------- 日间时段（6-18点）低段稳定点精细修正 ----------
    if low.any() and n >= 4:
        low_stable3 = low & (~strong_ramp)
        if low_stable3.any():
            daytime = low_stable3 & (hour >= 6.0) & (hour < 18.0)
            if daytime.any():
                out[daytime] = out[daytime] * 0.997

    # ---------- 低段极稳定点进一步压 ----------
    if n >= 5 and low.any():
        low_stable4 = low & (~strong_ramp)
        if low_stable4.any():
            range3 = np.zeros(n)
            if n >= 3:
                r = np.maximum(np.maximum(pp[:-2], pp[1:-1]), pp[2:]) - \
                    np.minimum(np.minimum(pp[:-2], pp[1:-1]), pp[2:])
                range3[1:-1] = r
            ultra_stable = low_stable4 & (range3 < 0.015 * cap) & (range3 > 0)
            if ultra_stable.any():
                out[ultra_stable] = out[ultra_stable] * 0.988

            night = low_stable4 & ((hour < 6.0) | (hour >= 18.0))
            if night.any():
                out[night] = out[night] * 0.994

    # ---------- 低段上沿过渡区温和上抬 ----------
    if low.any():
        low_stable5 = low & (~strong_ramp)
        if low_stable5.any():
            trans_zone = low_stable5 & (pp >= 0.18 * cap) & (pp < 0.20 * cap)
            if trans_zone.any():
                out[trans_zone] = out[trans_zone] * 1.008

    # ---------- 中段与低段交界处（20-22%Cap）稳定点温和校正 ----------
    if mid.any():
        mid_stable3 = mid & (~strong_ramp)
        if mid_stable3.any():
            mid_entry = mid_stable3 & (pp >= 0.20 * cap) & (pp < 0.22 * cap)
            if mid_entry.any():
                out[mid_entry] = out[mid_entry] * 0.988

    # ---------- 中段上沿（50-60%Cap）稳定点轻微压 ----------
    if mid.any():
        mid_stable4 = mid & (~strong_ramp)
        if mid_stable4.any():
            mid_upper = mid_stable4 & (pp >= 0.50 * cap) & (pp < 0.60 * cap)
            if mid_upper.any():
                out[mid_upper] = out[mid_upper] * 0.992

    # ---------- 低段强爬坡点更精细混合（低段爬坡占比较多） ----------
    if low.any():
        low_ramp2 = low & strong_ramp
        if low_ramp2.any():
            low_ramp_up = low_ramp2 & (pp >= 0.10 * cap)
            if low_ramp_up.any():
                out[low_ramp_up] = 0.88 * out[low_ramp_up] + 0.12 * med[low_ramp_up]

    # ---------- 新增：低段夜间极稳定点更精细收缩 ----------
    if n >= 5 and low.any():
        low_stable6 = low & (~strong_ramp)
        if low_stable6.any():
            night = low_stable6 & ((hour < 6.0) | (hour >= 18.0))
            night_low = night & (pp < 0.10 * cap)
            if night_low.any():
                out[night_low] = out[night_low] * 0.990
            night_deep = night & (pp < 0.05 * cap)
            if night_deep.any():
                out[night_deep] = out[night_deep] * 0.994

    # ---------- 新增：低段白天 midday 稳定点轻微收缩 ----------
    if n >= 4 and low.any():
        low_stable7 = low & (~strong_ramp)
        if low_stable7.any():
            midday = low_stable7 & (hour >= 10.0) & (hour < 15.0)
            if midday.any():
                out[midday] = out[midday] * 0.996

    # ---------- 新增：低段深夜（0-4点）超稳定收缩 ----------
    if n >= 5 and low.any():
        low_stable8 = low & (~strong_ramp)
        if low_stable8.any():
            deep_night = low_stable8 & (hour < 4.0)
            deep_night_vlow = deep_night & (pp < 0.08 * cap)
            if deep_night_vlow.any():
                out[deep_night_vlow] = out[deep_night_vlow] * 0.991

    # ---------- 新增：低段（8-15%Cap）稳定点二次收缩 ----------
    if n >= 3 and low.any():
        low_stable9 = low & (~strong_ramp)
        if low_stable9.any():
            band_0815 = low_stable9 & (pp >= 0.08 * cap) & (pp < 0.15 * cap)
            if band_0815.any():
                out[band_0815] = out[band_0815] * 0.993

    # ---------- 新增：低段稳定点整体再轻微收缩（主战场压制相对误差） ----------
    if n >= 3 and low.any():
        low_stable10 = low & (~strong_ramp)
        if low_stable10.any():
            out[low_stable10] = out[low_stable10] * 0.996

    out = np.clip(out, 0.0, cap)

    return out