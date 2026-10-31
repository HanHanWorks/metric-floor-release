import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 105.0
    n = len(pp)
    out = pp.copy()
    if n == 0:
        return out
    if n < 3:
        return np.clip(out, 0.0, cap)

    # 邻居与爬坡
    left = np.empty(n); right = np.empty(n)
    left[1:] = pp[:-1]; left[0] = pp[0]
    right[:-1] = pp[1:]; right[-1] = pp[-1]
    neigh = 0.5 * (left + right)
    dev = pp - neigh
    ramp_bwd = np.abs(pp - left)
    ramp_fwd = np.abs(right - pp)
    is_ramp = (ramp_fwd > 0.035 * cap) & (ramp_bwd > 0.035 * cap) & \
             (((right - pp) * (pp - left)) > 0)

    # 二阶差分（曲率），用于识别孤立尖刺/凹陷
    sec = left - 2.0 * pp + right

    # 1) 极低段：孤立尖刺向下修，孤立凹陷向上修（极保守）
    mask1 = (pp < 0.05 * cap) & ~is_ramp
    spike1 = mask1 & (pp > neigh + 0.025 * cap) & (left < 0.03 * cap) & (right < 0.03 * cap)
    out = np.where(spike1, 0.5 * pp + 0.5 * neigh, out)

    # 2) 低段：明显偏离去抖
    mask2 = (pp >= 0.05 * cap) & (pp < 0.15 * cap) & ~is_ramp
    adj2 = mask2 & (np.abs(dev) > 0.045 * cap)
    out = np.where(adj2, 0.88 * out + 0.12 * neigh, out)

    # 3) 中低段：温和去抖
    mask3 = (pp >= 0.15 * cap) & (pp < 0.40 * cap) & ~is_ramp
    adj3 = mask3 & (np.abs(dev) > 0.055 * cap)
    out = np.where(adj3, 0.93 * out + 0.07 * neigh, out)

    # 4) 中高段：抑制孤立尖峰/凹陷（用曲率增强判定）
    mask4 = (pp >= 0.40 * cap) & (pp <= 0.75 * cap) & ~is_ramp
    adj4 = mask4 & ((np.abs(dev) > 0.075 * cap) | (np.abs(sec) > 0.10 * cap))
    out = np.where(adj4, 0.92 * out + 0.08 * neigh, out)

    # 5) 高出力段：仅向下修极端孤立尖峰
    mask5 = (pp > 0.75 * cap) & (pp <= 0.90 * cap) & ~is_ramp
    adj5 = mask5 & (neigh < pp - 0.10 * cap)
    out = np.where(adj5, 0.82 * out + 0.18 * neigh, out)

    # 6) 极高出力段封顶
    out = np.where(pp > 0.97 * cap, np.minimum(out, 0.985 * cap), out)

    # 7) 跨日同时刻一致性：明显离群轻微修正
    if n >= 96:
        day_pts = 96
        prev_day = np.full(n, np.nan)
        next_day = np.full(n, np.nan)
        prev_day[day_pts:] = pp[:-day_pts]
        next_day[:-day_pts] = pp[day_pts:]
        valid = ~np.isnan(prev_day) | ~np.isnan(next_day)
        day_mean = np.where(
            ~np.isnan(prev_day) & ~np.isnan(next_day), 0.5 * (prev_day + next_day),
            np.where(~np.isnan(prev_day), prev_day, next_day)
        )
        mask7 = valid & ~is_ramp & (pp > 0.08 * cap) & (pp < 0.85 * cap)
        dev_d = np.abs(pp - day_mean)
        adj7 = mask7 & (dev_d > 0.16 * cap)
        out = np.where(adj7, 0.94 * out + 0.06 * day_mean, out)

    # 8) 爬坡段完全保护
    out = np.where(is_ramp, pp, out)

    # 9) 零值保护
    near_zero = pp < 0.03 * cap
    out = np.where(near_zero, np.minimum(out, pp + 0.006 * cap), out)

    # 10) 低出力平台期（连续低值）平滑
    low_ctx = (left < 0.10 * cap) & (right < 0.10 * cap) & (pp >= 0.03 * cap) & (pp < 0.10 * cap)
    mask10 = low_ctx & ~is_ramp & (np.abs(dev) > 0.02 * cap) & (np.abs(sec) > 0.03 * cap)
    out = np.where(mask10, 0.85 * out + 0.15 * neigh, out)

    # 11) 中段局部平台一致性
    mid_plateau = (pp >= 0.30 * cap) & (pp <= 0.45 * cap) & ~is_ramp
    adj11 = mid_plateau & (np.abs(sec) > 0.06 * cap) & (np.abs(dev) < 0.05 * cap)
    out = np.where(adj11, 0.95 * out + 0.05 * neigh, out)

    # 12) 低出力段（<0.20cap）连续两点同向偏离的联合抑制
    low_pair = (pp >= 0.03 * cap) & (pp < 0.20 * cap) & ~is_ramp
    pair_left_dev = np.empty(n); pair_left_dev[0] = 0.0
    pair_left_dev[1:] = pp[:-1] - neigh[:-1]
    pair_same = np.sign(dev) == np.sign(pair_left_dev)
    mask12 = low_pair & pair_same & (np.abs(dev) > 0.015 * cap) & (np.abs(dev) < 0.04 * cap)
    out = np.where(mask12, 0.90 * out + 0.10 * neigh, out)

    # 13) 高段（0.6-0.9cap）曲率异常但偏离不大的局部锯齿抑制
    high_plateau = (pp >= 0.55 * cap) & (pp <= 0.90 * cap) & ~is_ramp
    adj13 = high_plateau & (np.abs(sec) > 0.08 * cap) & (np.abs(dev) > 0.03 * cap)
    out = np.where(adj13, 0.94 * out + 0.06 * neigh, out)

    # 14) 低出力主导场景下宽窗三点对称平滑
    low_three = (pp >= 0.03 * cap) & (pp < 0.12 * cap) & \
                (left >= 0.02 * cap) & (right >= 0.02 * cap) & ~is_ramp
    three_mean = (left + pp + right) / 3.0
    adj14 = low_three & (np.abs(pp - three_mean) > 0.010 * cap) & \
            (np.abs(pp - three_mean) < 0.030 * cap)
    out = np.where(adj14, 0.80 * out + 0.20 * three_mean, out)

    # 15) 中低段连续同向小台阶的成对收缩
    midlow_pair = (pp >= 0.20 * cap) & (pp < 0.30 * cap) & ~is_ramp
    adj15 = midlow_pair & pair_same & (np.abs(dev) > 0.020 * cap) & (np.abs(dev) < 0.045 * cap)
    out = np.where(adj15, 0.93 * out + 0.07 * neigh, out)

    # 16) 新增：低出力段（0.03-0.12cap）平台稳定化——五点宽窗中位数附近收缩
    #     针对 frac_below_02cap≈53% 的数据画像，低段噪声主导，做更宽窗的稳健化
    five_lo = np.empty(n); five_lo[:] = pp
    for k in range(1, 3):
        shifted_l = np.empty(n); shifted_l[:] = pp
        shifted_l[k:] = pp[:-k]
        shifted_r = np.empty(n); shifted_r[:] = pp
        shifted_r[:-k] = pp[k:]
        five_lo = five_lo + shifted_l + shifted_r
    five_lo = five_lo / 5.0
    low_wide = (pp >= 0.04 * cap) & (pp < 0.12 * cap) & \
               (left >= 0.03 * cap) & (right >= 0.03 * cap) & ~is_ramp
    adj16 = low_wide & (np.abs(pp - five_lo) > 0.012 * cap) & \
            (np.abs(pp - five_lo) < 0.035 * cap)
    out = np.where(adj16, 0.85 * out + 0.15 * five_lo, out)

    # 17) 新增：中段（0.30-0.55cap）小曲率锯齿的一次轻平滑
    mid_saw = (pp >= 0.30 * cap) & (pp <= 0.55 * cap) & ~is_ramp
    adj17 = mid_saw & (np.abs(sec) > 0.045 * cap) & (np.abs(dev) > 0.020 * cap) & \
            (np.abs(dev) < 0.055 * cap)
    out = np.where(adj17, 0.95 * out + 0.05 * neigh, out)

    # 18) 边界裁剪
    out = np.clip(out, 0.0, cap)

    return out