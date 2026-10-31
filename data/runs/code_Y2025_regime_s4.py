import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 1.0
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    month = view['month'].values.astype(float)

    out = pp.copy()
    n = len(out)
    if n == 0:
        return out

    low = pp < 0.20 * cap
    mid = (pp >= 0.20 * cap) & (pp < 0.60 * cap)
    high = pp >= 0.60 * cap
    very_low = pp < 0.05 * cap
    low_not_very = low & (~very_low)

    diff = np.zeros(n)
    if n > 1:
        diff[1:] = pp[1:] - pp[:-1]
    ramp = np.abs(diff)
    strong_ramp = ramp > 0.10 * cap

    if n >= 3:
        sm = pp.copy()
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        sm[0] = 0.5 * pp[0] + 0.5 * pp[1]
        sm[-1] = 0.5 * pp[-1] + 0.5 * pp[-2]
        med = pp.copy()
        for i in range(1, n - 1):
            med[i] = np.median(pp[i-1:i+2])
        med[0] = np.median(pp[:min(3, n)])
        med[-1] = np.median(pp[max(0, n-3):])
    else:
        sm = pp.copy()
        med = pp.copy()
    blend = 0.5 * sm + 0.5 * med

    low_final = np.maximum(0.90 * blend, 0.0)

    mid_final = 0.6 * sm + 0.4 * med
    lag = 96
    if n > lag:
        mid_lag = mid_final.copy()
        mid_lag[lag:] = 0.88 * mid_final[lag:] + 0.12 * pp[:-lag]
        mid_final = mid_lag

    high_final = 0.7 * pp + 0.3 * blend

    out = np.where(low, low_final, out)
    out = np.where(mid, mid_final, out)
    out = np.where(high, high_final, out)

    if n >= 3:
        out = np.where(strong_ramp, 0.7 * blend + 0.3 * out, out)

    out = np.where(very_low, 0.95 * out, out)

    night = (hour < 6.0)
    dawn = (hour >= 6.0) & (hour < 9.0)
    out = np.where(low & night, 0.94 * out, out)
    out = np.where(low & dawn, 0.97 * out, out)

    weekend = (dow >= 5)
    winter = (month <= 2) | (month == 12)
    out = np.where(low & weekend, 0.97 * out, out)
    out = np.where(low & winter, 0.98 * out, out)

    if n > 1:
        down_ramp = diff < -0.02 * cap
        out = np.where(low & down_ramp, 0.97 * out, out)

    afternoon = (hour >= 12.0) & (hour < 16.0)
    out = np.where(low & afternoon, 0.97 * out, out)

    if n > 1:
        up_ramp = diff > 0.03 * cap
        out = np.where(low & up_ramp, 0.5 * out + 0.5 * pp, out)

    deep_night = (hour < 5.0) | (hour >= 21.0)
    out = np.where(low & deep_night, 0.92 * out, out)

    early_morn = (hour >= 5.0) & (hour < 7.0)
    out = np.where(low & early_morn, 0.96 * out, out)

    evening = (hour >= 17.0) & (hour < 20.0)
    out = np.where(low & evening, 0.96 * out, out)

    if n > 1:
        mid_up = mid & (diff > 0.03 * cap)
        out = np.where(mid_up, 0.6 * out + 0.4 * pp, out)

    very_low_night = very_low & ((hour < 6.0) | (hour >= 20.0))
    out = np.where(very_low_night, 0.90 * out, out)

    very_low_dawn = very_low & (hour >= 6.0) & (hour < 8.0)
    out = np.where(very_low_dawn, 1.02 * out, out)

    daytime = (hour >= 8.0) & (hour < 17.0)
    out = np.where(low_not_very & daytime, 0.985 * out, out)

    if n > 1:
        mid_down = mid & (diff < -0.03 * cap)
        out = np.where(mid_down, 0.97 * out, out)

    if n > 1:
        high_down = high & (diff < -0.03 * cap)
        out = np.where(high_down, 0.85 * out + 0.15 * pp, out)

    out = np.where(very_low, 0.97 * out, out)
    out = np.where(very_low & weekend, 0.96 * out, out)
    out = np.where(very_low & winter, 0.97 * out, out)

    mid_low_edge = (pp >= 0.20 * cap) & (pp < 0.30 * cap)
    out = np.where(mid_low_edge, 0.985 * out, out)
    mid_high_edge = (pp >= 0.50 * cap) & (pp < 0.60 * cap)
    out = np.where(mid_high_edge, 1.01 * out, out)

    if n > 1:
        out = np.where(very_low & (diff < -0.01 * cap), 0.96 * out, out)

    very_low_deep_night = very_low & ((hour < 4.0) | (hour >= 22.0))
    out = np.where(very_low_deep_night, 0.93 * out, out)

    out = np.where(low_not_very & ((hour < 6.0) | (hour >= 20.0)), 0.97 * out, out)
    out = np.where(mid, 0.99 * out, out)

    mid_core = (pp >= 0.30 * cap) & (pp < 0.50 * cap)
    out = np.where(mid_core, 1.005 * out, out)

    if n > 1:
        very_low_up = very_low & (diff > 0.01 * cap)
        out = np.where(very_low_up, 0.5 * out + 0.5 * pp, out)

    out = np.where(low_not_very, 0.985 * out, out)
    out = np.where(low_not_very & ((hour < 6.0) | (hour >= 20.0)), 0.975 * out, out)
    out = np.where(low_not_very & (hour >= 9.0) & (hour < 16.0), 0.985 * out, out)
    out = np.where(very_low & (hour >= 9.0) & (hour < 15.0), 0.96 * out, out)
    out = np.where(very_low & weekend, 0.97 * out, out)

    if n > 1:
        mid_down_night = mid & (diff < -0.02 * cap) & ((hour < 7.0) | (hour >= 19.0))
        out = np.where(mid_down_night, 0.97 * out, out)

    out = np.where(mid & (hour >= 9.0) & (hour < 16.0), 0.99 * out, out)
    out = np.where(high & (hour >= 10.0) & (hour < 14.0), 1.01 * out, out)

    if n >= 3:
        strong_ramp_mid = strong_ramp & mid
        out = np.where(strong_ramp_mid, 0.6 * blend + 0.4 * out, out)

    out = np.where(very_low & ((hour < 6.0) | (hour >= 21.0)), 0.95 * out, out)
    out = np.where(very_low & (hour >= 10.0) & (hour < 14.0), 0.97 * out, out)
    out = np.where(very_low & winter, 0.97 * out, out)
    out = np.where(low_not_very & (hour >= 17.0) & (hour < 20.0), 0.98 * out, out)
    out = np.where(low_not_very & weekend & (hour >= 9.0) & (hour < 16.0), 0.98 * out, out)

    out = np.where(very_low & ((hour < 5.0) | (hour >= 22.0)), 0.96 * out, out)
    out = np.where(very_low & (hour >= 5.0) & (hour < 8.0), 0.97 * out, out)
    out = np.where(very_low & (hour >= 17.0) & (hour < 21.0), 0.97 * out, out)
    out = np.where(very_low, 0.985 * out, out)
    out = np.where(low_not_very, 0.99 * out, out)

    vl_deepest = very_low & ((hour < 4.0) | (hour >= 22.0))
    out = np.where(vl_deepest, 0.97 * out, out)
    out = np.where(very_low & weekend & ((hour < 6.0) | (hour >= 20.0)), 0.97 * out, out)
    out = np.where(very_low & winter & ((hour < 6.0) | (hour >= 20.0)), 0.97 * out, out)
    out = np.where(very_low & (hour >= 12.0) & (hour < 16.0), 0.98 * out, out)
    out = np.where(very_low, 0.99 * out, out)
    out = np.where(low_not_very, 0.995 * out, out)

    out = np.where(mid_core, 1.003 * out, out)
    out = np.where(mid & ((hour < 6.0) | (hour >= 21.0)), 0.985 * out, out)
    if n > 1:
        mid_strong_up = mid & (diff > 0.05 * cap)
        out = np.where(mid_strong_up, 0.5 * out + 0.5 * pp, out)

    out = np.where(high, 0.85 * out + 0.15 * pp, out)

    out = np.where(very_low, 0.96 * out, out)
    out = np.where(low_not_very & (hour >= 10.0) & (hour < 15.0), 0.98 * out, out)
    out = np.where(very_low & ((hour < 4.0) | (hour >= 22.0)), 0.97 * out, out)
    out = np.where(very_low & weekend & (hour >= 9.0) & (hour < 16.0), 0.97 * out, out)

    if n > 1:
        mid_strong_ramp = mid & (ramp > 0.08 * cap)
        out = np.where(mid_strong_ramp, 0.55 * out + 0.45 * pp, out)
    if n > 1:
        mid_strong_down = mid & (diff < -0.05 * cap)
        out = np.where(mid_strong_down, 0.95 * out, out)

    out = np.where(high, 0.92 * out + 0.08 * pp, out)
    if n > 1:
        high_up = high & (diff > 0.05 * cap)
        out = np.where(high_up, 0.7 * out + 0.3 * pp, out)

    out = np.where(very_low & ((hour < 3.0) | (hour >= 23.0)), 0.97 * out, out)
    out = np.where(very_low & winter & ((hour < 6.0) | (hour >= 20.0)), 0.98 * out, out)
    out = np.where(very_low, 0.99 * out, out)
    out = np.where(low_not_very & (hour >= 11.0) & (hour < 14.0), 0.99 * out, out)

    if n > 1:
        mid_up_strong = mid & (diff > 0.10 * cap)
        out = np.where(mid_up_strong, 0.5 * out + 0.5 * pp, out)

    out = np.where(high, 0.95 * out + 0.05 * pp, out)

    out = np.where(very_low & (hour >= 4.0) & (hour < 6.0), 0.98 * out, out)
    out = np.where(low_not_very & (hour >= 6.0) & (hour < 9.0), 0.99 * out, out)
    out = np.where(very_low & (hour >= 18.0) & (hour < 21.0), 0.985 * out, out)
    if n > 1:
        mid_flat = mid & (ramp < 0.02 * cap)
        out = np.where(mid_flat, 0.995 * out, out)

    out = np.where(very_low & (hour >= 9.0) & (hour < 16.0), 0.985 * out, out)
    out = np.where(low_not_very & (hour >= 12.0) & (hour < 15.0), 0.99 * out, out)
    out = np.where(very_low & ((hour < 6.0) | (hour >= 20.0)), 0.99 * out, out)
    out = np.where(mid, 0.997 * out, out)

    out = np.where(very_low, 0.985 * out, out)
    out = np.where(low_not_very, 0.992 * out, out)
    out = np.where(very_low & ((hour < 7.0) | (hour >= 19.0)), 0.975 * out, out)
    out = np.where(mid & ((hour < 6.0) | (hour >= 20.0)), 0.995 * out, out)

    # 本轮新增：低段强爬坡段向平滑靠拢（强爬坡平滑更稳），极低出力再轻压
    if n > 1:
        low_ramp = low & strong_ramp
        out = np.where(low_ramp, 0.55 * out + 0.45 * blend, out)
    # 低出力段整体再向零微压（该段占比最高、相对误差主导）
    out = np.where(low, 0.997 * out, out)
    # 中出力段强爬坡保留更多原始，避免过度平滑导致跟踪滞后
    if n > 1:
        mid_ramp = mid & strong_ramp
        out = np.where(mid_ramp, 0.55 * out + 0.45 * pp, out)

    out = np.clip(out, 0.0, cap)
    return out