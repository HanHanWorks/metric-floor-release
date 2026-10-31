import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view else 89.125
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float) if 'dow' in view else np.zeros(len(pp))
    month = view['month'].values.astype(float) if 'month' in view else np.ones(len(pp))
    n = len(pp)
    cap = max(cap, 1e-6)

    out = pp.copy()
    ratio = pp / cap

    # 1) 低功率段主校准：低段收缩（约 68% 落在 0.2cap 以下）
    low_w = np.clip(1.0 - ratio / 0.35, 0.0, 1.0)
    out = out * (1.0 - 0.15 * low_w)

    # 2) 极低值去噪
    low_mask = out < 0.02 * cap
    if low_mask.any() and n >= 5:
        pad = np.pad(out, (2, 2), mode='edge')
        local_mean = np.convolve(pad, np.ones(5) / 5.0, mode='valid')
        kill = low_mask & (local_mean < 0.03 * cap)
        out = np.where(kill, out * 0.62, out)

    # 3) 夜间收缩（夜间普遍偏低，抑制残余偏高）
    night_mask = (hour < 5.0) | (hour >= 22.0)
    if night_mask.any():
        night_med = np.median(out[night_mask])
        excess = out - night_med
        shrink = np.where(night_mask & (excess > 0.05 * cap), 0.91, 1.0)
        out = out * shrink

    # 4) 高功率段抑制
    high_mask = out > 0.88 * cap
    if high_mask.any():
        out = np.where(high_mask, 0.88 * cap + (out - 0.88 * cap) * 0.82, out)

    # 5) 自适应平滑（保留 ramp 特征）
    if n >= 3:
        pad = np.pad(out, (1, 1), mode='edge')
        smooth = (pad[:-2] + 2.0 * pad[1:-1] + pad[2:]) / 4.0
        dev = np.abs(out - smooth)
        blend = np.clip(dev / (0.05 * cap + 1e-6), 0.0, 1.0) * 0.26
        out = out * (1.0 - blend) + smooth * blend

    # 6) 分位数分段收缩
    r = out / cap
    out = np.where((r > 0.03) & (r <= 0.30), out * 0.960, out)
    out = np.where((r > 0.30) & (r <= 0.55), out * 0.983, out)
    out = np.where((r > 0.55) & (r <= 0.75), out * 0.993, out)

    # 7) ramp 约束
    if n >= 2:
        max_ramp = 0.07 * cap
        diff = np.diff(out)
        big = np.abs(diff) > max_ramp
        if big.any():
            clipped = np.clip(diff, -max_ramp, max_ramp)
            new_out = out.copy()
            new_out[1:] = new_out[0] + np.cumsum(clipped)
            idx = np.where(big)[0] + 1
            out[idx] = new_out[idx]

    # 8) 按小时中位数做日内校准
    if n >= 96:
        hour_int = np.floor(hour).astype(int) % 24
        hour_med = np.zeros(24)
        for h in range(24):
            m = hour_int == h
            if m.any():
                hour_med[h] = np.median(out[m])
        for h in range(24):
            m = hour_int == h
            if m.any() and hour_med[h] > 1e-6:
                vals = out[m]
                excess = vals - hour_med[h]
                adj = np.where(excess > 0.07 * cap, vals - excess * 0.10, vals)
                out[m] = adj

    # 9) 整体校准（略强）
    out = out * 0.972

    # 10) 低功率平台压降
    r2 = out / cap
    out = np.where(r2 < 0.10, out * 0.955, out)
    out = np.where((r2 >= 0.10) & (r2 < 0.20), out * 0.977, out)
    out = np.where((r2 >= 0.20) & (r2 < 0.30), out * 0.987, out)

    # 11) 夜间极低段再压
    r3 = out / cap
    out = np.where(r3 < 0.05, out * 0.970, out)

    # 12) 尖峰残差抑制
    if n >= 5:
        pad = np.pad(out, (2, 2), mode='edge')
        med5 = np.median(np.stack([pad[i:i+n] for i in range(5)], axis=0), axis=0)
        dev = out - med5
        spike = np.abs(dev) > 0.09 * cap
        if spike.any():
            out = np.where(spike, med5 + dev * 0.72, out)

    # 13) 季节/星期微调
    weekend = (dow >= 5.0)
    if weekend.any():
        rw = out / cap
        out = np.where(weekend & (rw > 0.10) & (rw < 0.45), out * 0.994, out)

    winter = (month == 12.0) | (month == 1.0) | (month == 2.0)
    if winter.any():
        rw2 = out / cap
        out = np.where(winter & (rw2 > 0.15) & (rw2 < 0.45), out * 0.996, out)

    # 14) 额外整体微调
    out = out * 0.995

    # 15) 非负裁剪 + 上限
    out = np.clip(out, 0.0, cap)

    # 常数兜底
    if np.std(out) < 1e-9:
        out = pp.copy()

    return out