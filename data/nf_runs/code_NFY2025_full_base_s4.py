import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float).copy()
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 93.625
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # --- 轻度中值滤波 (窗口3) ---
    xp = np.pad(x, 1, mode='reflect')
    xm = np.empty_like(x)
    for i in range(n):
        xm[i] = np.median(xp[i:i+3])
    x = xm

    # --- 温和平滑 (5点加权) ---
    if n >= 5:
        out = x.copy()
        out[2:-2] = (0.1*x[:-4] + 0.2*x[1:-3] + 0.4*x[2:-2]
                     + 0.2*x[3:-1] + 0.1*x[4:])
        x = out

    # --- 低出力区轻微收缩 ---
    low_thr = 0.08 * cap
    mask_low = x < low_thr
    x[mask_low] = x[mask_low] * 0.93

    # --- 极低值抬底 ---
    floor = 0.004 * cap
    x = np.where(x < floor, x * 0.6 + floor * 0.4, x)

    # --- 爬坡限制 ---
    max_ramp = 0.07 * cap
    for i in range(1, n):
        d = x[i] - x[i-1]
        if d > max_ramp:
            x[i] = x[i-1] + max_ramp
        elif d < -max_ramp:
            x[i] = x[i-1] - max_ramp

    # --- 滞后混合 ---
    alpha = 0.10
    x_lag = x.copy()
    x_lag[1:] = (1-alpha)*x[1:] + alpha*x[:-1]
    x = x_lag

    # --- 容量约束 ---
    x = np.clip(x, 0.0, cap)

    # --- 日历修正 ---
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        day_mask = (hour >= 8) & (hour <= 17)
        night_mask = (hour < 5) | (hour > 20)
        x[day_mask] = x[day_mask] * 1.015
        x[night_mask] = x[night_mask] * 0.97
        x = np.clip(x, 0.0, cap)

    # --- 分段映射: 高值区温和压缩 ---
    hi_thr = 0.60 * cap
    mask_hi = x > hi_thr
    x[mask_hi] = hi_thr + (x[mask_hi] - hi_thr) * 0.90

    # --- 分位拉伸: 中段轻微上抬 ---
    mid_thr = 0.15 * cap
    mask_mid = (x > mid_thr) & (x <= hi_thr)
    x[mask_mid] = x[mask_mid] * 1.01

    # --- 极低值段轻微收缩 ---
    vlow_thr = 0.15 * cap
    tiny_thr = 0.02 * cap
    mask_vlow = (x > tiny_thr) & (x < vlow_thr)
    x[mask_vlow] = x[mask_vlow] * 0.985

    # --- 轻量二阶平滑，仅用于平缓段 ---
    if n >= 5:
        y = x.copy()
        sm = (0.15*x[:-4] + 0.2*x[1:-3] + 0.3*x[2:-2]
              + 0.2*x[3:-1] + 0.15*x[4:])
        local_ramp = np.abs(np.diff(x, prepend=x[0]))
        ramp_slice = local_ramp[2:-2]
        calm = ramp_slice < (0.02 * cap)
        idx = np.arange(2, n-2)
        y[idx[calm]] = sm[calm]
        x = y

    # --- 夜间低出力段再温和压低 ---
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        deep_night = (hour < 5) | (hour > 22)
        low_m = deep_night & (x < 0.10 * cap)
        x[low_m] = x[low_m] * 0.95

    # --- 日间一致性修正 ---
    if n >= 96:
        day_len = 96
        num_days = n // day_len
        if num_days >= 2:
            daily_mean = np.array([
                x[d*day_len:(d+1)*day_len].mean() for d in range(num_days)
            ])
            overall = daily_mean.mean()
            if overall > 1e-6:
                for d in range(num_days):
                    ratio = daily_mean[d] / overall
                    s = d * day_len
                    e = s + day_len
                    if ratio < 0.7:
                        x[s:e] = x[s:e] * 0.99
                    elif ratio > 1.3:
                        x[s:e] = x[s:e] * 1.005

    # --- 低出力微调 ---
    lowband_m = (x > 0.02 * cap) & (x < 0.10 * cap)
    x[lowband_m] = x[lowband_m] * 0.99

    # --- 日周期形态校正 ---
    if 'hour' in view.columns and n >= 96:
        hour = view['hour'].values.astype(float)
        core_day = (hour >= 10) & (hour <= 15)
        trans = ((hour >= 6) & (hour < 8)) | ((hour > 17) & (hour <= 20))
        if core_day.any():
            x[core_day] = x[core_day] * 1.004
        if trans.any():
            x[trans] = x[trans] * 0.995

    # --- 极端高值再压缩 ---
    vhi_thr = 0.80 * cap
    mask_vhi = x > vhi_thr
    x[mask_vhi] = vhi_thr + (x[mask_vhi] - vhi_thr) * 0.85

    # --- 极低出力轻微再收缩 ---
    lowzero = x < 0.015 * cap
    x[lowzero] = x[lowzero] * 0.96

    # === 精修 (保留原有) ===
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        peak_mask = (hour >= 11) & (hour <= 14) & (x > 0.5 * cap)
        if peak_mask.any():
            x[peak_mask] = x[peak_mask] * 1.006

    if n >= 3:
        prev_v = x[:-2]
        next_v = x[2:]
        cur = x[1:-1]
        neigh_mean = 0.5 * (prev_v + next_v)
        spike = cur > (neigh_mean + 0.05 * cap)
        dip = cur < (neigh_mean - 0.05 * cap)
        x[1:-1][spike] = 0.75 * cur[spike] + 0.25 * neigh_mean[spike]
        x[1:-1][dip] = 0.85 * cur[dip] + 0.15 * neigh_mean[dip]

    vlow2 = (x > 0.005 * cap) & (x < 0.05 * cap)
    x[vlow2] = x[vlow2] * 0.985

    # === 新增: 尾段与削峰协同精修 ===
    # (D) 顶部 5% 区域再轻压，配合考核口径对高估更敏感的特性
    top_thr = 0.85 * cap
    top_m = x > top_thr
    x[top_m] = top_thr + (x[top_m] - top_thr) * 0.90

    # (E) 中间高值带 (0.45~0.65cap) 温和上抬，弥补前期多重收缩的过度抑制
    band_m = (x > 0.45 * cap) & (x <= 0.65 * cap)
    x[band_m] = x[band_m] * 1.008

    # (F) 极低出力 (<0.03cap) 二次收缩，抑制夜间误报
    verylow_m = (x > 0.0) & (x < 0.03 * cap)
    x[verylow_m] = x[verylow_m] * 0.98

    # (G) 日间核心峰值段相位对齐微调
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        ph = (hour >= 12) & (hour < 14) & (x > 0.55 * cap)
        if ph.any():
            x[ph] = x[ph] * 1.004

    # (H) 爬坡后骤降点温和抬升，减少对下降沿的低估
    if n >= 3:
        d_prev = x[1:-1] - x[:-2]
        d_next = x[2:] - x[1:-1]
        drop = (d_prev > 0.02 * cap) & (d_next < -0.03 * cap)
        x[1:-1][drop] = x[1:-1][drop] * 1.01

    x = np.clip(x, 0.0, cap)
    x = np.maximum(x, 0.0)

    return x