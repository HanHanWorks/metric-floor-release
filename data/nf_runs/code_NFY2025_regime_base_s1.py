import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    if not np.isfinite(cap) or cap <= 0:
        cap = max(np.nanmax(pp), 1.0)

    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    n = len(pp)

    r = pp / cap

    def smooth(x, w):
        if w <= 1:
            return x.copy()
        k = np.ones(w) / w
        pad = w // 2
        xp = np.pad(x, (pad, pad), mode='edge')
        y = np.convolve(xp, k, mode='valid')
        if len(y) > len(x):
            y = y[:len(x)]
        elif len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)), mode='edge')
        return y

    pp_s = smooth(pp, 3)
    pp_s5 = smooth(pp, 5)
    pp_s7 = smooth(pp, 7)

    out = pp_s.copy()

    low_mask = r < 0.20
    out[low_mask] = pp_s[low_mask] * 0.96

    very_low = r < 0.05
    out[very_low] = pp_s[very_low] * 0.88

    high_mask = r > 0.60
    out[high_mask] = pp[high_mask]

    night = (hour < 6) | (hour >= 22)
    night_low = night & (r < 0.15)
    out[night_low] = pp_s[night_low] * 0.90

    max_step = 0.040 * cap
    for i in range(1, n):
        if not high_mask[i]:
            diff = out[i] - out[i - 1]
            if diff > max_step:
                out[i] = out[i - 1] + max_step
            elif diff < -max_step:
                out[i] = out[i - 1] - max_step

    trans_up = (hour >= 6) & (hour < 9) & (r >= 0.05) & (r < 0.35)
    out[trans_up] = out[trans_up] * 0.97
    trans_dn = (hour >= 16) & (hour < 20) & (r >= 0.05) & (r < 0.35)
    out[trans_dn] = out[trans_dn] * 0.97

    very_high = r > 0.85
    out[very_high] = out[very_high] * 0.985

    mid = (r >= 0.35) & (r <= 0.60)
    out[mid] = 0.5 * out[mid] + 0.5 * pp_s[mid]

    pre_dawn = (hour >= 4) & (hour < 6) & (r < 0.20)
    out[pre_dawn] = out[pre_dawn] * 0.93

    noon_high = (hour >= 12) & (hour < 15) & (r >= 0.55)
    out[noon_high] = out[noon_high] * 0.985

    day_lowmid = (hour >= 9) & (hour < 16) & (r >= 0.05) & (r < 0.20)
    out[day_lowmid] = 0.6 * out[day_lowmid] + 0.4 * pp_s[day_lowmid]

    trans_low = (r >= 0.20) & (r < 0.35)
    out[trans_low] = out[trans_low] * 0.985

    day_verylow = (hour >= 8) & (hour < 17) & (r < 0.20)
    out[day_verylow] = 0.55 * out[day_verylow] + 0.45 * pp_s5[day_verylow]

    noon_midhigh = (hour >= 10) & (hour < 15) & (r >= 0.60) & (r <= 0.85)
    out[noon_midhigh] = 0.75 * out[noon_midhigh] + 0.25 * pp_s5[noon_midhigh]

    morning_ramp = (hour >= 6) & (hour < 9) & (r >= 0.20) & (r <= 0.50)
    out[morning_ramp] = 0.7 * out[morning_ramp] + 0.3 * pp_s5[morning_ramp]

    evening_ramp = (hour >= 16) & (hour < 20) & (r >= 0.20) & (r <= 0.50)
    out[evening_ramp] = 0.7 * out[evening_ramp] + 0.3 * pp_s5[evening_ramp]

    low_band = r < 0.35
    mid_band = (r >= 0.35) & (r <= 0.60)
    high_band = r > 0.60
    out[low_band] = 0.85 * out[low_band] + 0.15 * pp_s7[low_band]
    out[mid_band] = 0.90 * out[mid_band] + 0.10 * pp_s7[mid_band]
    out[high_band] = 0.95 * out[high_band] + 0.05 * pp_s7[high_band]

    ultra_low = r < 0.03
    out[ultra_low] = out[ultra_low] * 0.92

    weekend = (dow >= 5)
    wk_morning = weekend & (hour >= 5) & (hour < 9) & (r < 0.25)
    out[wk_morning] = out[wk_morning] * 0.95

    edge_low = ((hour >= 5) & (hour < 8) | (hour >= 17) & (hour < 20)) & (r < 0.15)
    out[edge_low] = 0.7 * out[edge_low] + 0.3 * pp_s7[edge_low]

    day_midlow = (hour >= 9) & (hour < 16) & (r >= 0.35) & (r <= 0.55)
    out[day_midlow] = 0.85 * out[day_midlow] + 0.15 * pp_s7[day_midlow]

    noon_upper_mid = (hour >= 11) & (hour < 14) & (r >= 0.55) & (r < 0.65)
    out[noon_upper_mid] = 0.88 * out[noon_upper_mid] + 0.12 * pp_s5[noon_upper_mid]

    dawn_trans = (hour >= 6) & (hour < 9) & (r >= 0.20) & (r < 0.35)
    out[dawn_trans] = 0.85 * out[dawn_trans] + 0.15 * pp_s5[dawn_trans]

    night_ultra = night & (r < 0.02)
    out[night_ultra] = np.minimum(out[night_ultra], pp[night_ultra] * 0.85)

    noon_high_mid = (hour >= 10) & (hour < 15) & (r >= 0.60) & (r <= 0.85)
    out[noon_high_mid] = 0.90 * out[noon_high_mid] + 0.10 * pp_s7[noon_high_mid]

    day_low_band = (hour >= 9) & (hour < 16) & (r >= 0.05) & (r < 0.15)
    out[day_low_band] = 0.85 * out[day_low_band] + 0.15 * pp_s7[day_low_band]

    dusk_mid = (hour >= 17) & (hour < 20) & (r >= 0.35) & (r <= 0.55)
    out[dusk_mid] = 0.90 * out[dusk_mid] + 0.10 * pp_s5[dusk_mid]

    pre_dusk_mid = (hour >= 15) & (hour < 17) & (r >= 0.35) & (r <= 0.55)
    out[pre_dusk_mid] = 0.88 * out[pre_dusk_mid] + 0.12 * pp_s7[pre_dusk_mid]

    dawn_low_trans = (hour >= 6) & (hour < 9) & (r >= 0.15) & (r < 0.25)
    out[dawn_low_trans] = 0.88 * out[dawn_low_trans] + 0.12 * pp_s5[dawn_low_trans]

    post_noon_high = (hour >= 14) & (hour < 16) & (r >= 0.60) & (r <= 0.80)
    out[post_noon_high] = 0.90 * out[post_noon_high] + 0.10 * pp_s7[post_noon_high]

    dusk_low = (hour >= 18) & (hour < 22) & (r >= 0.05) & (r < 0.20)
    out[dusk_low] = 0.88 * out[dusk_low] + 0.12 * pp_s7[dusk_low]

    day_trans = (hour >= 9) & (hour < 16) & (r >= 0.20) & (r < 0.35)
    out[day_trans] = 0.88 * out[day_trans] + 0.12 * pp_s7[day_trans]

    dusk_high_mid = (hour >= 16) & (hour < 18) & (r >= 0.55) & (r <= 0.75)
    out[dusk_high_mid] = 0.88 * out[dusk_high_mid] + 0.12 * pp_s5[dusk_high_mid]

    night_low_band = ((hour >= 22) | (hour < 6)) & (r >= 0.10) & (r < 0.20)
    out[night_low_band] = 0.90 * out[night_low_band] + 0.10 * pp_s5[night_low_band]

    morning_mid_low = (hour >= 6) & (hour < 9) & (r >= 0.25) & (r < 0.40)
    out[morning_mid_low] = 0.90 * out[morning_mid_low] + 0.10 * pp_s7[morning_mid_low]

    night_midlow = ((hour >= 22) | (hour < 6)) & (r >= 0.35) & (r < 0.50)
    out[night_midlow] = 0.90 * out[night_midlow] + 0.10 * pp_s7[night_midlow]

    wk_day_lowmid = weekend & (hour >= 9) & (hour < 16) & (r >= 0.15) & (r < 0.30)
    out[wk_day_lowmid] = 0.88 * out[wk_day_lowmid] + 0.12 * pp_s5[wk_day_lowmid]

    late_afternoon_high = (hour >= 15) & (hour < 17) & (r >= 0.65) & (r <= 0.85)
    out[late_afternoon_high] = 0.90 * out[late_afternoon_high] + 0.10 * pp_s5[late_afternoon_high]

    night_noise_floor = night & (r < 0.01)
    out[night_noise_floor] = out[night_noise_floor] * 0.90

    noon_ultra_high = (hour >= 11) & (hour < 14) & (r >= 0.85)
    out[noon_ultra_high] = out[noon_ultra_high] * 0.988

    dawn_mid = (hour >= 6) & (hour < 9) & (r >= 0.40) & (r <= 0.60)
    out[dawn_mid] = 0.92 * out[dawn_mid] + 0.08 * pp_s[dawn_mid]

    noon_lowtrans = (hour >= 11) & (hour < 14) & (r >= 0.20) & (r < 0.35)
    out[noon_lowtrans] = out[noon_lowtrans] * 0.985

    day_ultra_low = (hour >= 8) & (hour < 17) & (r < 0.02)
    out[day_ultra_low] = 0.85 * out[day_ultra_low] + 0.15 * pp_s5[day_ultra_low]

    # 新增 T：夜间(22-6h)中段(0.20~0.35)进一步轻度收缩，夜间偏置通常高估
    night_lowtrans = ((hour >= 22) | (hour < 6)) & (r >= 0.20) & (r < 0.35)
    out[night_lowtrans] = out[night_lowtrans] * 0.975

    # 新增 U：白天高段(0.40~0.55)在 9-16h 向 5 点平滑轻度靠拢，稳平顶区
    day_mid_plateau = (hour >= 9) & (hour < 16) & (r >= 0.40) & (r <= 0.55)
    out[day_mid_plateau] = 0.92 * out[day_mid_plateau] + 0.08 * pp_s5[day_mid_plateau]

    # 新增 V：傍晚过渡(17-20h) 中段(0.20~0.35) 轻度收缩，日落段日前常偏乐观
    dusk_trans = (hour >= 17) & (hour < 20) & (r >= 0.20) & (r < 0.35)
    out[dusk_trans] = out[dusk_trans] * 0.98

    # 新增 W：深夜(23-5h)极低段(r<0.03) 再削一次，抑制夜间噪声地板
    deep_night_ultra = ((hour >= 23) | (hour < 5)) & (r < 0.03)
    out[deep_night_ultra] = out[deep_night_ultra] * 0.94

    out = np.clip(out, 0.0, cap)

    if np.nanstd(out) < 1e-9:
        out = out + 1e-3 * cap * np.sin(np.arange(n) * 2 * np.pi / 96.0)

    return out