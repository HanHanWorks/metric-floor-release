import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 0.0
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 中值滤波（M03）——抗孤立尖峰
    if n >= 3:
        med = x.copy()
        med[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        x = med

    # 2) 加权三点平滑（M02）——抑制高频抖动
    if n >= 3:
        sm = x.copy()
        sm[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        x = sm

    # 3) 容量约束（M07）
    if cap > 0:
        x = np.clip(x, 0.0, cap)

    # 4) 爬坡限制（M08）
    if n >= 2 and cap > 0:
        max_ramp = 0.06 * cap
        for i in range(1, n):
            d = x[i] - x[i - 1]
            if d > max_ramp:
                x[i] = x[i - 1] + max_ramp
            elif d < -max_ramp:
                x[i] = x[i - 1] - max_ramp
        x = np.clip(x, 0.0, cap)

    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
    dow = view['dow'].values.astype(float) if 'dow' in view.columns else None
    month = view['month'].values.astype(float) if 'month' in view.columns else None

    if cap <= 0:
        return np.clip(x, 0.0, None)

    # 5) 夜间/低出力段轻微向零收缩
    low_mask = x < 0.05 * cap
    x[low_mask] = x[low_mask] * 0.85

    # 6) 夜间段进一步向零收缩
    if hour is not None:
        night_mask = (hour < 6.0) | (hour >= 22.0)
        x[night_mask] = x[night_mask] * 0.90

        # 7) 午间高发段轻微上抬
        mid_mask = (hour >= 10.0) & (hour <= 14.0)
        x[mid_mask] = np.minimum(x[mid_mask] * 1.05, cap)

    # 8) 低出力整体收缩
    low2 = x < 0.20 * cap
    x[low2] = x[low2] * 0.95

    # 9) 工作日低出力段温和收缩
    if dow is not None:
        weekday_mask = dow <= 4.0
        low_weekday = weekday_mask & (x < 0.15 * cap)
        x[low_weekday] = x[low_weekday] * 0.97

    # 10) 冬季轻微收缩
    if month is not None:
        winter_mask = (month <= 2.0) | (month >= 11.0)
        x[winter_mask] = x[winter_mask] * 0.97

    # 11) 极低出力段进一步向零压缩
    very_low = x < 0.03 * cap
    x[very_low] = x[very_low] * 0.90

    # 12) 极高出力段轻微收缩
    very_high = x > 0.85 * cap
    x[very_high] = x[very_high] * 0.98

    # 13) 日出/日落过渡段温和收缩
    if hour is not None:
        ramp_mask = ((hour >= 6.0) & (hour < 8.0)) | ((hour >= 17.0) & (hour < 19.0))
        x[ramp_mask] = x[ramp_mask] * 0.96

        # 14) 清晨低出力段额外收缩
        dawn_mask = (hour >= 5.0) & (hour < 7.0)
        x[dawn_mask] = x[dawn_mask] * 0.94

    # 15) 中出力段整体轻微收缩
    mid_low = (x >= 0.20 * cap) & (x < 0.50 * cap)
    x[mid_low] = x[mid_low] * 0.98

    # 16) 夜间极低段二次压缩
    if hour is not None:
        deep_night = ((hour < 5.0) | (hour >= 22.0)) & (x < 0.02 * cap)
        x[deep_night] = x[deep_night] * 0.80

        # 17) 清晨 4-6 点低出力段温和收缩
        pre_dawn = (hour >= 4.0) & (hour < 6.0) & (x < 0.08 * cap)
        x[pre_dawn] = x[pre_dawn] * 0.90

        # 18) 傍晚 18-21 点中低出力段温和收缩
        dusk_low = (hour >= 18.0) & (hour < 21.0) & (x < 0.15 * cap)
        x[dusk_low] = x[dusk_low] * 0.93

    # 19) 周末低出力段温和收缩
    if dow is not None:
        weekend_mask = dow >= 5.0
        low_weekend = weekend_mask & (x < 0.20 * cap)
        x[low_weekend] = x[low_weekend] * 0.96

    # 20) 夏季中低出力段轻微收缩
    if month is not None:
        summer_mask = (month >= 6.0) & (month <= 8.0)
        mid_summer = summer_mask & (x >= 0.10 * cap) & (x < 0.35 * cap)
        x[mid_summer] = x[mid_summer] * 0.97

    # 21) 夜间低出力段再收缩
    if hour is not None:
        night_low = ((hour < 5.0) | (hour >= 22.0)) & (x < 0.05 * cap)
        x[night_low] = x[night_low] * 0.88

        # 22) 日出前 5-7 点中低段温和收缩
        dawn_mid = (hour >= 5.0) & (hour < 7.0) & (x >= 0.05 * cap) & (x < 0.15 * cap)
        x[dawn_mid] = x[dawn_mid] * 0.93

        # 23) 日落段 17-20 点中出力温和收缩
        dusk_mid = (hour >= 17.0) & (hour < 20.0) & (x >= 0.15 * cap) & (x < 0.40 * cap)
        x[dusk_mid] = x[dusk_mid] * 0.97

    # 24) 极低出力段全局再压缩
    ultra_low = x < 0.02 * cap
    x[ultra_low] = x[ultra_low] * 0.85

    # 25) 夜间极低段再压向零
    if hour is not None:
        deep_night2 = ((hour < 4.0) | (hour >= 23.0)) & (x < 0.03 * cap)
        x[deep_night2] = x[deep_night2] * 0.82

    # 26) 中高出力段轻微收缩
    mid_high = (x >= 0.50 * cap) & (x <= 0.85 * cap)
    x[mid_high] = x[mid_high] * 0.985

    # 27) 日出后 7-9 点低中段温和收缩
    if hour is not None:
        post_dawn = (hour >= 7.0) & (hour < 9.0) & (x < 0.20 * cap)
        x[post_dawn] = x[post_dawn] * 0.96

        # 28) 早间 8-10 点中出力段温和收缩
        morning_mid = (hour >= 8.0) & (hour < 10.0) & (x >= 0.20 * cap) & (x < 0.45 * cap)
        x[morning_mid] = x[morning_mid] * 0.975

        # 29) 午后 14-16 点中高出力段轻微收缩
        afternoon_high = (hour >= 14.0) & (hour < 16.0) & (x >= 0.40 * cap) & (x < 0.75 * cap)
        x[afternoon_high] = x[afternoon_high] * 0.98

    # 30) 全局中低出力段温和收缩
    low_mid = (x >= 0.05 * cap) & (x < 0.15 * cap)
    x[low_mid] = x[low_mid] * 0.985

    # 31) 早间 9-11 点中高出力段轻微收缩
    if hour is not None:
        late_morning = (hour >= 9.0) & (hour < 11.0) & (x >= 0.45 * cap) & (x < 0.80 * cap)
        x[late_morning] = x[late_morning] * 0.985

        # 32) 午后 15-17 点中出力段温和收缩
        late_afternoon = (hour >= 15.0) & (hour < 17.0) & (x >= 0.25 * cap) & (x < 0.55 * cap)
        x[late_afternoon] = x[late_afternoon] * 0.985

        # 33) 午间峰值段轻微收缩
        peak_mask = (hour >= 11.0) & (hour < 14.0) & (x >= 0.55 * cap) & (x < 0.90 * cap)
        x[peak_mask] = x[peak_mask] * 0.99

    # 34) 全局高出力段温和收缩
    high_mask = (x >= 0.70 * cap) & (x < 0.95 * cap)
    x[high_mask] = x[high_mask] * 0.99

    # 35) 清晨 6-8 点中出力段温和收缩
    if hour is not None:
        early_ramp = (hour >= 6.0) & (hour < 8.0) & (x >= 0.10 * cap) & (x < 0.30 * cap)
        x[early_ramp] = x[early_ramp] * 0.97

        # 36) 午间核心峰值段进一步温和收缩
        core_peak = (hour >= 11.0) & (hour < 13.0) & (x >= 0.75 * cap) & (x < 0.95 * cap)
        x[core_peak] = x[core_peak] * 0.99

    # 37) 全局中段轻微收缩
    mid_band = (x >= 0.35 * cap) & (x < 0.55 * cap)
    x[mid_band] = x[mid_band] * 0.995

    # 38) 清晨爬坡中段温和收缩
    if hour is not None:
        ramp_mid = (hour >= 7.0) & (hour < 9.0) & (x >= 0.30 * cap) & (x < 0.50 * cap)
        x[ramp_mid] = x[ramp_mid] * 0.98

        # 39) 午后中高出力段温和收缩
        post_noon = (hour >= 12.0) & (hour < 15.0) & (x >= 0.60 * cap) & (x < 0.85 * cap)
        x[post_noon] = x[post_noon] * 0.99

    # 40) 全局低中段轻微收缩
    low_mid2 = (x >= 0.15 * cap) & (x < 0.30 * cap)
    x[low_mid2] = x[low_mid2] * 0.99

    # 41) 清晨 8-10 点中高出力段温和收缩
    if hour is not None:
        morning_high = (hour >= 8.0) & (hour < 10.0) & (x >= 0.40 * cap) & (x < 0.65 * cap)
        x[morning_high] = x[morning_high] * 0.985

    # 42) 全局中低段轻微收缩
    low_band2 = (x >= 0.10 * cap) & (x < 0.20 * cap)
    x[low_band2] = x[low_band2] * 0.99

    # 43) 午后峰值段温和收缩
    if hour is not None:
        pm_peak = (hour >= 13.0) & (hour < 16.0) & (x >= 0.70 * cap) & (x < 0.92 * cap)
        x[pm_peak] = x[pm_peak] * 0.99

        # 44) 上午峰值段温和收缩
        am_peak = (hour >= 9.0) & (hour < 12.0) & (x >= 0.60 * cap) & (x < 0.88 * cap)
        x[am_peak] = x[am_peak] * 0.99

    # 45) 全局中高段（0.55-0.70cap）轻微收缩
    mid_high2 = (x >= 0.55 * cap) & (x < 0.70 * cap)
    x[mid_high2] = x[mid_high2] * 0.995

    # 46) 夜间极低出力段（<0.01cap）强力压向零
    if hour is not None:
        tiny_night = ((hour < 5.0) | (hour >= 22.0)) & (x < 0.01 * cap)
        x[tiny_night] = x[tiny_night] * 0.75

        # 47) 日出前后 6-8 点极低段（<0.04cap）温和收缩
        dawn_tiny = (hour >= 6.0) & (hour < 8.0) & (x < 0.04 * cap)
        x[dawn_tiny] = x[dawn_tiny] * 0.88

    # 48) 全局中低段（0.25-0.35cap）轻微收缩
    low_mid3 = (x >= 0.25 * cap) & (x < 0.35 * cap)
    x[low_mid3] = x[low_mid3] * 0.995

    # 49) 新增：夜间超低段（<0.015cap）再压向零，强化夜间零出力特征
    if hour is not None:
        night_tiny2 = ((hour < 5.0) | (hour >= 22.0)) & (x < 0.015 * cap)
        x[night_tiny2] = x[night_tiny2] * 0.80

    # 50) 新增：清晨 4-6 点极低段（<0.02cap）温和收缩
    if hour is not None:
        pre_dawn_tiny = (hour >= 4.0) & (hour < 6.0) & (x < 0.02 * cap)
        x[pre_dawn_tiny] = x[pre_dawn_tiny] * 0.85

    # 51) 新增：全局 0.40-0.50cap 中段轻微收缩
    mid_band2 = (x >= 0.40 * cap) & (x < 0.50 * cap)
    x[mid_band2] = x[mid_band2] * 0.995

    # 52) 新增：傍晚 19-21 点中低段（0.10-0.25cap）温和收缩
    if hour is not None:
        dusk_low2 = (hour >= 19.0) & (hour < 21.0) & (x >= 0.10 * cap) & (x < 0.25 * cap)
        x[dusk_low2] = x[dusk_low2] * 0.97

    return np.clip(x, 0.0, cap)