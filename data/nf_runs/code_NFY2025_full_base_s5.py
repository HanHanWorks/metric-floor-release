import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view else 142.2
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # M02 平滑：轻度移动平均（保留边缘）
    k = 3
    if n >= k:
        kernel = np.ones(k) / k
        pad = k // 2
        xp = np.pad(x, pad, mode='edge')
        x = np.convolve(xp, kernel, mode='valid')[:n]

    # M09 滞后混合：与前一时刻做轻度混合，抑制抖动
    if n > 1:
        prev = np.concatenate([[x[0]], x[:-1]])
        x = 0.85 * x + 0.15 * prev

    # M12 分段映射：对中高出力段做轻微压缩，避免过冲
    mid_mask = (x >= 0.10 * cap) & (x < 0.50 * cap)
    x[mid_mask] = x[mid_mask] * 0.97
    high_mask = x >= 0.50 * cap
    x[high_mask] = x[high_mask] * 0.95

    # 抬底：合并为简洁分段抬升
    low2 = x < 0.05 * cap
    x[low2] = x[low2] + 0.005 * cap

    low3 = (x >= 0.05 * cap) & (x < 0.12 * cap)
    x[low3] = x[low3] + 0.003 * cap

    low4 = x < 0.02 * cap
    x[low4] = x[low4] + 0.002 * cap

    low5 = x < 0.03 * cap
    x[low5] = x[low5] + 0.0015 * cap

    low6 = (x >= 0.12 * cap) & (x < 0.18 * cap)
    x[low6] = x[low6] + 0.001 * cap

    midhigh = (x >= 0.18 * cap) & (x < 0.35 * cap)
    x[midhigh] = x[midhigh] + 0.0015 * cap

    high2 = (x >= 0.35 * cap) & (x < 0.55 * cap)
    x[high2] = x[high2] + 0.0012 * cap

    low7 = x < 0.015 * cap
    x[low7] = x[low7] + 0.0012 * cap

    low8 = (x >= 0.03 * cap) & (x < 0.08 * cap)
    x[low8] = x[low8] + 0.0015 * cap

    low9 = (x >= 0.08 * cap) & (x < 0.10 * cap)
    x[low9] = x[low9] + 0.001 * cap

    low10 = (x >= 0.02 * cap) & (x < 0.04 * cap)
    x[low10] = x[low10] + 0.0012 * cap

    low11 = (x >= 0.10 * cap) & (x < 0.14 * cap)
    x[low11] = x[low11] + 0.0008 * cap

    # 日历修正
    if 'hour' in view:
        hour = view['hour'].values.astype(float)

        morning = (hour >= 6.5) & (hour < 9.5)
        evening = (hour >= 16.5) & (hour < 20.5)
        noon = (hour >= 11.0) & (hour < 14.0)
        x[morning] = x[morning] + 0.0025 * cap
        x[evening] = x[evening] + 0.0025 * cap
        x[noon] = x[noon] - 0.0015 * cap

        night = (hour >= 0.0) & (hour < 5.0)
        x[night] = x[night] + 0.0018 * cap

        dusk = (hour >= 20.5) & (hour < 22.0)
        x[dusk] = x[dusk] + 0.0015 * cap

        pre_morning = (hour >= 5.0) & (hour < 6.5)
        x[pre_morning] = x[pre_morning] + 0.0015 * cap

        late_evening = (hour >= 22.0) & (hour < 23.5)
        x[late_evening] = x[late_evening] + 0.001 * cap

        sunrise = (hour >= 6.0) & (hour < 7.0)
        x[sunrise] = x[sunrise] + 0.001 * cap

        post_noon = (hour >= 14.0) & (hour < 15.0)
        x[post_noon] = x[post_noon] + 0.0008 * cap

        daytime = (hour >= 8.0) & (hour < 18.0)
        x[daytime] = x[daytime] + 0.0006 * cap

        if 'month' in view:
            month = view['month'].values.astype(int)
            summer_noon = ((month == 6) | (month == 7) | (month == 8))
            sn_mask = summer_noon & (hour >= 11.0) & (hour < 14.0)
            x[sn_mask] = x[sn_mask] - 0.0008 * cap

    if 'dow' in view:
        dow = view['dow'].values.astype(int)
        weekend = (dow == 5) | (dow == 6)
        x[weekend] = x[weekend] + 0.0012 * cap

    if 'month' in view:
        month = view['month'].values.astype(int)
        winter = (month == 12) | (month == 1) | (month == 2)
        x[winter] = x[winter] + 0.0008 * cap

    # M10 分位拉伸：轻微抬升中低分位段，改善欠预测（保守幅度）
    q_low = (x >= 0.05 * cap) & (x < 0.25 * cap)
    x[q_low] = x[q_low] * 1.015 + 0.001 * cap

    q_mid = (x >= 0.25 * cap) & (x < 0.45 * cap)
    x[q_mid] = x[q_mid] * 1.01

    # 容量约束
    x = np.clip(x, 0.0, cap)

    return x