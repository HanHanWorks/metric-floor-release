import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    month = view['month'].values.astype(float) if 'month' in view.columns else np.full(len(pp), 6.0)
    n = len(pp)

    if not (np.isfinite(cap) and cap > 0):
        m = np.nanmax(pp) if np.isfinite(np.nanmax(pp)) and np.nanmax(pp) > 0 else 1.0
        cap = m

    pp_s = pp.copy()
    pp_s = np.where(np.isfinite(pp_s), pp_s, 0.0)
    pp_s = np.clip(pp_s, 0.0, cap)

    # 1) 极轻平滑去毛刺
    if n >= 3:
        kernel = np.array([0.05, 0.90, 0.05])
        pp_sm = np.convolve(pp_s, kernel, mode='same')
        pp_sm[0] = pp_s[0]
        pp_sm[-1] = pp_s[-1]
        pp_s = pp_sm

    # 2) 分段收缩：低出力段收缩更细，整体略收
    ratio = pp_s / cap

    mask = ratio < 0.03
    pp_s = np.where(mask, pp_s * 0.88, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.03) & (ratio < 0.08)
    pp_s = np.where(mask, pp_s * 0.855, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.08) & (ratio < 0.15)
    pp_s = np.where(mask, pp_s * 0.875, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.15) & (ratio < 0.22)
    pp_s = np.where(mask, pp_s * 0.895, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.22) & (ratio < 0.35)
    pp_s = np.where(mask, pp_s * 0.945, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.35) & (ratio < 0.55)
    pp_s = np.where(mask, pp_s * 0.978, pp_s)
    ratio = pp_s / cap

    mask = (ratio >= 0.55) & (ratio < 0.75)
    pp_s = np.where(mask, pp_s * 0.992, pp_s)
    ratio = pp_s / cap

    mask = ratio >= 0.75
    pp_s = np.where(mask, pp_s * 0.996, pp_s)

    # 3) 爬坡限制：平滑自适应，夜间允许更小爬坡
    base_ramp = 0.045 * cap
    for i in range(1, n):
        h = hour[i]
        if h < 5 or h >= 21:
            mr = base_ramp * 0.7
        elif (h >= 5) & (h < 7):
            mr = base_ramp * 0.9
        elif (h >= 10) & (h <= 14):
            mr = base_ramp * 1.1
        else:
            mr = base_ramp
        d = pp_s[i] - pp_s[i - 1]
        if d > mr:
            pp_s[i] = pp_s[i - 1] + mr
        elif d < -mr:
            pp_s[i] = pp_s[i - 1] - mr

    # 4) 时段形状校正（略保守）
    hour_mod = np.ones(n)
    night = (hour < 5) | (hour >= 21)
    dawn = (hour >= 5) & (hour < 7)
    dusk = (hour >= 18) & (hour < 21)
    noon = (hour >= 10) & (hour <= 14)
    shoulder = ((hour >= 7) & (hour < 10)) | ((hour > 14) & (hour < 18))
    hour_mod[night] = 0.84
    hour_mod[dawn] = 0.93
    hour_mod[dusk] = 0.94
    hour_mod[noon] = 1.035
    hour_mod[shoulder] = 1.000
    pp_s = pp_s * hour_mod

    # 5) 周末温和收缩
    weekend = (dow >= 5)
    pp_s = np.where(weekend, pp_s * 0.95, pp_s)

    # 6) 季节调制：夏季午间略增，冬季整体略降
    season_mod = np.ones(n)
    summer = (month >= 6) & (month <= 8)
    winter = (month == 12) | (month <= 2)
    spring_fall = ~(summer | winter)
    season_mod[summer] = 1.015
    season_mod[winter] = 0.97
    season_mod[spring_fall] = 0.995
    summer_noon = summer & noon
    season_mod[summer_noon] = 1.035
    pp_s = pp_s * season_mod

    # 7) 物理约束
    pp_s = np.clip(pp_s, 0.0, cap)

    # 8) 全局温和收缩（保守，避免过冲）
    pp_s = pp_s * 0.972

    # 9) 最终物理约束
    pp_s = np.clip(pp_s, 0.0, cap)

    # 10) 保底
    if n > 1 and np.allclose(pp_s, pp_s[0]):
        pp_s = pp_s + 0.001 * cap * np.sin(2 * np.pi * hour / 24.0)

    return pp_s