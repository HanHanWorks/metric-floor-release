import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = view['Cap'].values.astype(float)
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    month = view['month'].values.astype(float)

    n = len(pp)
    cap_safe = np.where(cap > 0, cap, 1.0)
    ratio = pp / cap_safe

    # 数据画像：pp_mean_ratio≈0.20，frac_below_01cap≈0.47，step_autocorr≈0.9988，
    # ramp_p99≈0.043。序列极平滑。ACC 已在 59.4 附近进入平台期。
    # 本轮策略：在保持历史有效修正的基础上，做更细粒度的分段偏差修正，
    # 重点压低整体正偏差累积，同时保护低出力区绝对误差与爬坡平滑。
    # 关键调整：
    #   - 低出力区收缩略减（避免过度压低导致欠拟合）
    #   - 中低区收缩略增（压制正偏差）
    #   - 极低区抬升保持
    #   - ramp 平滑权重微调，减少对正常波动干扰

    out_ratio = ratio.copy()

    # 1) 低出力区连续收缩：低端约 10%，随 ratio 平滑衰减到 0.25 处为 0
    low_w = np.clip((0.25 - ratio) / 0.25, 0.0, 1.0)
    low_w = np.sqrt(low_w)
    low_shrink = 1.0 - 0.10 * low_w
    out_ratio = out_ratio * low_shrink

    # 2) 夜间平滑收缩（diurnal 权重，夜间≈1，白天≈0）
    hour_frac = (hour % 24.0) / 24.0
    diurnal = 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour_frac - 0.5)))
    night_weight = np.clip(1.0 - diurnal / 0.15, 0.0, 1.0)
    night_shrink = 1.0 - 0.075 * night_weight
    out_ratio = out_ratio * night_shrink

    # 3) 高出力区几乎不动
    high_mask = ratio > 0.80
    out_ratio[high_mask] = ratio[high_mask] * 0.996
    very_high_mask = ratio > 0.90
    out_ratio[very_high_mask] = ratio[very_high_mask] * 0.998

    # 4) 极低出力温和抬升，减小绝对误差
    very_low_mask = ratio < 0.02
    if np.any(very_low_mask):
        out_ratio[very_low_mask] = out_ratio[very_low_mask] + 0.0045
    low_band_mask = (ratio >= 0.02) & (ratio < 0.05)
    if np.any(low_band_mask):
        band_w = (0.05 - ratio[low_band_mask]) / 0.03
        out_ratio[low_band_mask] = out_ratio[low_band_mask] + 0.0026 * band_w

    # 5) 中低出力区收缩（0.05-0.25），压制正偏差累积
    mid_low_mask = (ratio >= 0.05) & (ratio < 0.25)
    if np.any(mid_low_mask):
        mid_w = (0.25 - ratio[mid_low_mask]) / 0.20
        out_ratio[mid_low_mask] = out_ratio[mid_low_mask] * (1.0 - 0.0130 * mid_w)

    # 5b) 0.25-0.50 区间收缩
    mid_mask = (ratio >= 0.25) & (ratio < 0.50)
    if np.any(mid_mask):
        mid2_w = (0.50 - ratio[mid_mask]) / 0.25
        out_ratio[mid_mask] = out_ratio[mid_mask] * (1.0 - 0.0065 * mid2_w)

    # 5c) 0.50-0.80 区间极轻微收缩
    mid_high_mask = (ratio >= 0.50) & (ratio <= 0.80)
    if np.any(mid_high_mask):
        out_ratio[mid_high_mask] = out_ratio[mid_high_mask] * 0.9985

    # 6) 基于 ramp 的温和平滑
    if n > 2:
        dr = np.abs(np.diff(ratio, prepend=ratio[0]))
        ramp_thresh = 0.035
        ramp_mask = dr > ramp_thresh
        if np.any(ramp_mask):
            ratio_prev = np.roll(ratio, 1)
            ratio_next = np.roll(ratio, -1)
            ratio_prev[0] = ratio[0]
            ratio_next[-1] = ratio[-1]
            smooth_val = 0.80 * ratio + 0.10 * ratio_prev + 0.10 * ratio_next
            out_ratio[ramp_mask] = 0.55 * out_ratio[ramp_mask] + 0.45 * smooth_val[ramp_mask]

    # 7) 限制范围
    out_ratio = np.clip(out_ratio, 0.0, 1.0)

    out = out_ratio * cap_safe

    if n > 1 and np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out