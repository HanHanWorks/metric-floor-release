import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view else 105.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    if n >= 3:
        med = out.copy()
        med[1:-1] = np.median(np.stack([out[:-2], out[1:-1], out[2:]], axis=0), axis=0)
        out = med

    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    low_mask = out < 0.05 * cap
    out[low_mask] = out[low_mask] * 0.90

    out = np.clip(out, 0.0, cap)

    ramp_lim = 0.08 * cap
    for i in range(1, n):
        d = out[i] - out[i-1]
        if d > ramp_lim:
            out[i] = out[i-1] + ramp_lim
        elif d < -ramp_lim:
            out[i] = out[i-1] - ramp_lim

    out = out * 1.03
    out = np.clip(out, 0.0, cap)

    ratio = out / cap
    low_seg = ratio < 0.10
    out[low_seg] = out[low_seg] * 1.05
    high_seg = ratio > 0.60
    out[high_seg] = out[high_seg] * 0.97

    floor = 0.008 * cap
    nz = out > 0.0
    out[nz] = np.maximum(out[nz], floor)
    out = np.clip(out, 0.0, cap)

    if n >= 2:
        lag = out.copy()
        lag[1:] = 0.85 * out[1:] + 0.15 * out[:-1]
        out = lag

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        night = (hour < 6.0)
        out[night] = out[night] * 1.02
        noon = (hour >= 10.0) & (hour < 14.0)
        out[noon] = out[noon] * 0.985

    ratio = out / cap
    mid = (ratio >= 0.15) & (ratio < 0.45)
    out[mid] = out[mid] * 1.015
    midhi = (ratio >= 0.45) & (ratio <= 0.60)
    out[midhi] = out[midhi] * 0.99
    out = np.clip(out, 0.0, cap)

    if 'dow' in view and n >= 96:
        days = n // 96
        if days >= 1:
            reshaped = out[:days*96].reshape(days, 96)
            day_mean = reshaped.mean(axis=1, keepdims=True)
            dev = reshaped - day_mean
            thresh = 0.25 * cap
            mask = np.abs(dev) > thresh
            reshaped = np.where(mask, reshaped - 0.05 * dev, reshaped)
            out[:days*96] = reshaped.reshape(-1)
    out = np.clip(out, 0.0, cap)

    ratio = out / cap
    very_low = ratio < 0.03
    out[very_low] = out[very_low] * 1.02
    low_mid = (ratio >= 0.03) & (ratio < 0.12)
    out[low_mid] = out[low_mid] * 1.01

    if n >= 3:
        for i in range(1, n-1):
            if out[i-1] > 0.01 * cap and out[i] < 0.005 * cap and out[i+1] > 0.01 * cap:
                out[i] = 0.5 * (out[i-1] + out[i+1])
    out = np.clip(out, 0.0, cap)

    if n >= 96:
        days = n // 96
        if days >= 1:
            reshaped = out[:days*96].reshape(days, 96)
            win = 8
            anchor = np.zeros_like(reshaped)
            for d in range(days):
                row = reshaped[d]
                for j in range(0, 96, win):
                    seg = row[j:j+win]
                    anchor[d, j:j+win] = np.median(seg)
            dev = reshaped - anchor
            big = np.abs(dev) > 0.30 * cap
            reshaped = np.where(big, reshaped - 0.08 * dev, reshaped)
            out[:days*96] = reshaped.reshape(-1)

    ratio = out / cap
    tiny = (ratio > 0.0) & (ratio < 0.02)
    out[tiny] = out[tiny] * 1.03

    if n >= 96:
        days = n // 96
        if days >= 1:
            reshaped = out[:days*96].reshape(days, 96)
            for d in range(days):
                row = reshaped[d]
                dmax = row.max()
                if dmax > 0.05 * cap:
                    hi = row > 0.80 * dmax
                    row[hi] = row[hi] * 0.995
                    midm = (row >= 0.30 * dmax) & (row <= 0.60 * dmax)
                    row[midm] = row[midm] * 1.008
                reshaped[d] = row
            out[:days*96] = reshaped.reshape(-1)

    if n >= 192:
        days = n // 96
        if days >= 2:
            reshaped = out[:days*96].reshape(days, 96)
            for d in range(1, days):
                reshaped[d] = 0.94 * reshaped[d] + 0.06 * reshaped[d-1]
            out[:days*96] = reshaped.reshape(-1)
    out = np.clip(out, 0.0, cap)

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        dawn = (hour >= 6.0) & (hour < 9.0) & (ratio > 0.02) & (ratio < 0.35)
        out[dawn] = out[dawn] * 1.012
        dusk = (hour >= 16.0) & (hour < 19.0) & (ratio > 0.10)
        out[dusk] = out[dusk] * 0.992
        deepnight = ((hour >= 22.0) | (hour < 5.0)) & (ratio < 0.05)
        out[deepnight] = out[deepnight] * 0.97

    ratio = out / cap
    midhigh = (ratio >= 0.40) & (ratio <= 0.70)
    if midhigh.sum() >= 3:
        idx = np.where(midhigh)[0]
        smoothed = out.copy()
        for i in idx:
            if 0 < i < n-1:
                smoothed[i] = 0.85 * out[i] + 0.075 * out[i-1] + 0.075 * out[i+1]
        out = smoothed

    ratio = out / cap
    lowband = (ratio > 0.02) & (ratio < 0.15)
    out[lowband] = out[lowband] * 1.008

    ratio = out / cap
    veryhigh = ratio > 0.85
    out[veryhigh] = out[veryhigh] * 0.985

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        night_tiny = ((hour >= 21.0) | (hour < 5.0)) & (ratio > 0.0) & (ratio < 0.01)
        out[night_tiny] = out[night_tiny] * 0.90

    if n >= 3:
        diff = np.abs(np.diff(out))
        big_jump = diff > 0.06 * cap
        if big_jump.any():
            idxs = np.where(big_jump)[0]
            for i in idxs:
                if 0 <= i < n-1:
                    avg = 0.5 * (out[i] + out[i+1])
                    out[i] = 0.7 * out[i] + 0.3 * avg
                    out[i+1] = 0.7 * out[i+1] + 0.3 * avg

    ratio = out / cap
    lowmid2 = (ratio >= 0.15) & (ratio < 0.30)
    out[lowmid2] = out[lowmid2] * 1.006
    midlow = (ratio >= 0.30) & (ratio < 0.45)
    out[midlow] = out[midlow] * 1.004
    highband = (ratio >= 0.70) & (ratio <= 0.85)
    out[highband] = out[highband] * 0.995

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        night_low = (hour < 5.0) & (ratio > 0.0) & (ratio < 0.05)
        out[night_low] = out[night_low] * 0.95
        day_mid = (hour >= 9.0) & (hour < 15.0) & (ratio >= 0.20) & (ratio < 0.55)
        out[day_mid] = out[day_mid] * 1.005
    out = np.clip(out, 0.0, cap)

    ratio = out / cap
    band_010_020 = (ratio >= 0.10) & (ratio < 0.20)
    out[band_010_020] = out[band_010_020] * 1.004
    band_045_055 = (ratio >= 0.45) & (ratio < 0.55)
    out[band_045_055] = out[band_045_055] * 1.003
    band_090_100 = (ratio >= 0.90) & (ratio <= 1.0)
    out[band_090_100] = out[band_090_100] * 0.992

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        dawn_ramp = (hour >= 5.0) & (hour < 7.0) & (ratio > 0.0) & (ratio < 0.10)
        out[dawn_ramp] = out[dawn_ramp] * 1.01
        dusk_band = (hour >= 18.0) & (hour < 20.0) & (ratio > 0.03) & (ratio < 0.15)
        out[dusk_band] = out[dusk_band] * 0.99
    out = np.clip(out, 0.0, cap)

    if n >= 96:
        days = n // 96
        if days >= 1:
            reshaped = out[:days*96].reshape(days, 96)
            for d in range(days):
                row = reshaped[d]
                row_med = np.median(row)
                if row_med > 0.02 * cap:
                    dev = row - row_med
                    big_dev = np.abs(dev) > 0.35 * cap
                    if big_dev.any():
                        row = np.where(big_dev, row - 0.06 * dev, row)
                reshaped[d] = row
            out[:days*96] = reshaped.reshape(-1)

    ratio = out / cap
    tiny_nz = (ratio >= 0.005) & (ratio < 0.02)
    out[tiny_nz] = out[tiny_nz] * 1.015
    band_055_070 = (ratio >= 0.55) & (ratio < 0.70)
    out[band_055_070] = out[band_055_070] * 0.996

    # ===== 本轮新增：更精细的分段与时段校准 =====
    ratio = out / cap
    core = (ratio >= 0.20) & (ratio < 0.50)
    out[core] = out[core] * 1.0025

    ratio = out / cap
    hi_band = (ratio >= 0.60) & (ratio < 0.85)
    out[hi_band] = out[hi_band] * 0.9975

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        daytime = (hour >= 8.0) & (hour < 17.0) & (ratio > 0.05) & (ratio < 0.85)
        out[daytime] = out[daytime] * 1.003

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        night_band = (hour < 6.0) & (ratio > 0.0) & (ratio < 0.08)
        out[night_band] = out[night_band] * 0.99

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        aft_hi = (hour >= 14.0) & (hour < 17.0) & (ratio > 0.55) & (ratio < 0.90)
        out[aft_hi] = out[aft_hi] * 0.996

    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        morning_ramp = (hour >= 7.0) & (hour < 10.0) & (ratio > 0.08) & (ratio < 0.40)
        out[morning_ramp] = out[morning_ramp] * 1.004

    out = np.clip(out, 0.0, cap)

    # ===== 本轮新增：基于数据画像的针对性微调 =====
    # 画像显示 pp_p50 约 0.19cap，frac_below_02cap 约 53%，主能量集中在中低段
    # 42) 中低段(0.10-0.25)温和抬升，这是最高频区间
    ratio = out / cap
    freq_band = (ratio >= 0.10) & (ratio < 0.25)
    out[freq_band] = out[freq_band] * 1.003

    # 43) 极低出力(0.02-0.08)轻抬，减少低估
    ratio = out / cap
    verylow_band = (ratio >= 0.02) & (ratio < 0.08)
    out[verylow_band] = out[verylow_band] * 1.006

    # 44) 峰值段(0.75-0.90)轻压，抑制高估
    ratio = out / cap
    peak_band = (ratio >= 0.75) & (ratio < 0.90)
    out[peak_band] = out[peak_band] * 0.996

    # 45) 午间核心(11-14)中高段轻压，抑制午间过估
    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        noon_core = (hour >= 11.0) & (hour < 14.0) & (ratio >= 0.40) & (ratio < 0.75)
        out[noon_core] = out[noon_core] * 0.997

    # 46) 傍晚(17-20)低段轻压，日落段易高估
    if 'hour' in view:
        hour = view['hour'].values.astype(float)
        ratio = out / cap
        eve_low = (hour >= 17.0) & (hour < 20.0) & (ratio > 0.0) & (ratio < 0.12)
        out[eve_low] = out[eve_low] * 0.99

    # 47) 全局轻微上调，配合前面多段抬升的累积效应
    ratio = out / cap
    overall_mid = (ratio >= 0.08) & (ratio < 0.65)
    out[overall_mid] = out[overall_mid] * 1.001

    out = np.clip(out, 0.0, cap)

    return out