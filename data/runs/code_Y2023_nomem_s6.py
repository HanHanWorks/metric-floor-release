import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 74.375
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    cap_safe = max(cap, 1e-6)

    # 1) 轻度平滑（保留爬坡），低出力段权重更大
    if n >= 5:
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        padded = np.pad(pp, 2, mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        ratio = pp / cap_safe
        w = np.clip(1.0 - ratio / 0.5, 0.0, 1.0) * 0.5
        out = (1 - w) * pp + w * smooth

    # 2) 夜间/零出力修正
    small_thresh = 0.02 * cap
    small_mask = out < small_thresh
    out[small_mask] = out[small_mask] * 0.7

    # 3) 极端高出力轻微压缩
    high_thresh = 0.9 * cap
    high_mask = out > high_thresh
    out[high_mask] = high_thresh + (out[high_mask] - high_thresh) * 0.85

    out = np.clip(out, 0.0, cap)

    # 5) 日周期一致性（局部中值回拉）
    if n >= 9:
        padded = np.pad(out, 4, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, 9)
        med = np.median(windows, axis=1)
        dev = out - med
        thresh = 0.08 * cap
        big = np.abs(dev) > thresh
        out[big] = out[big] - 0.15 * dev[big]
        out = np.clip(out, 0.0, cap)

    # 6) 出力上限软约束
    soft_high = 0.75 * cap
    m = out > soft_high
    out[m] = soft_high + (out[m] - soft_high) * 0.92
    out = np.clip(out, 0.0, cap)

    # 7) 分段轻度收缩
    ratio = out / cap_safe
    shrink = np.where(ratio < 0.1, 1.0,
             np.where(ratio < 0.5, 0.985,
             np.where(ratio < 0.75, 0.97, 0.95)))
    out = out * shrink
    out = np.clip(out, 0.0, cap)

    # 8) 保守偏置
    ratio2 = out / cap_safe
    bias = np.where(ratio2 < 0.05, 1.0,
           np.where(ratio2 < 0.2, 0.995,
           np.where(ratio2 < 0.6, 0.99, 0.985)))
    out = out * bias

    # 9) 分位数分段收缩
    ratio3 = out / cap_safe
    seg = np.where(ratio3 < 0.15, 1.0,
          np.where(ratio3 < 0.50, 0.992,
          np.where(ratio3 < 0.70, 0.978, 0.962)))
    out = out * seg

    # 10) 夜间时段压低
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night = (hour < 6.0) | (hour >= 20.0)
        night_ratio = out / cap_safe
        night_mask = night & (night_ratio > 0.0) & (night_ratio < 0.15)
        out[night_mask] = out[night_mask] * 0.93
        out = np.clip(out, 0.0, cap)

    # 11) 爬坡抑制
    if n >= 3:
        ramp_thresh = 0.06 * cap
        d = np.diff(out)
        big_up = d > ramp_thresh
        big_dn = d < -ramp_thresh
        if np.any(big_up):
            idx = np.where(big_up)[0] + 1
            out[idx] = out[idx - 1] + ramp_thresh + (out[idx] - out[idx - 1] - ramp_thresh) * 0.6
        if np.any(big_dn):
            idx = np.where(big_dn)[0] + 1
            out[idx] = out[idx - 1] - ramp_thresh + (out[idx] - out[idx - 1] + ramp_thresh) * 0.6
        out = np.clip(out, 0.0, cap)

    # 12-13) 低/中低出力段收缩
    ratio4 = out / cap_safe
    low_mask = (ratio4 >= 0.02) & (ratio4 < 0.10)
    out[low_mask] = out[low_mask] * 0.975
    mid_low_mask = (ratio4 >= 0.10) & (ratio4 < 0.20)
    out[mid_low_mask] = out[mid_low_mask] * 0.99

    # 14) 正午时段轻微上抬
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        noon = (hour >= 10.0) & (hour <= 14.0)
        noon_ratio = out / cap_safe
        noon_mask = noon & (noon_ratio > 0.3) & (noon_ratio < 0.85)
        out[noon_mask] = out[noon_mask] * 1.008
        out = np.clip(out, 0.0, cap)

    # 15) 极低出力向零收缩
    ratio5 = out / cap_safe
    tiny_mask = (ratio5 > 0.0) & (ratio5 < 0.02)
    out[tiny_mask] = out[tiny_mask] * 0.90

    # 16) 中高出力段收缩
    mid_high_mask = (ratio5 >= 0.50) & (ratio5 < 0.70)
    out[mid_high_mask] = out[mid_high_mask] * 0.995

    # 17) 清晨过渡收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        morning = (hour >= 6.0) & (hour < 9.0)
        mr = out / cap_safe
        mm = morning & (mr > 0.05) & (mr < 0.35)
        out[mm] = out[mm] * 0.985
        out = np.clip(out, 0.0, cap)

    # 19) 傍晚过渡收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        evening = (hour >= 17.0) & (hour < 20.0)
        er = out / cap_safe
        em = evening & (er > 0.05) & (er < 0.30)
        out[em] = out[em] * 0.99
        out = np.clip(out, 0.0, cap)

    # 20) 局部一致性回拉
    if n >= 5:
        p3 = np.pad(out, 1, mode='edge')
        local3 = (p3[:-2] + p3[1:-1] + p3[2:]) / 3.0
        out = 0.97 * out + 0.03 * local3
        out = np.clip(out, 0.0, cap)

    # 21-22) 中段/高段收缩
    ratio6 = out / cap_safe
    mid_mask = (ratio6 >= 0.20) & (ratio6 < 0.35)
    out[mid_mask] = out[mid_mask] * 0.997
    high2_mask = (ratio6 >= 0.70) & (ratio6 < 0.90)
    out[high2_mask] = out[high2_mask] * 0.99

    # 23) 细爬坡软约束
    if n >= 2:
        ramp_soft = 0.035 * cap
        d2 = np.diff(out)
        up2 = d2 > ramp_soft
        dn2 = d2 < -ramp_soft
        if np.any(up2):
            idx = np.where(up2)[0] + 1
            out[idx] = out[idx - 1] + ramp_soft + (out[idx] - out[idx - 1] - ramp_soft) * 0.85
        if np.any(dn2):
            idx = np.where(dn2)[0] + 1
            out[idx] = out[idx - 1] - ramp_soft + (out[idx] - out[idx - 1] + ramp_soft) * 0.85
        out = np.clip(out, 0.0, cap)

    # 24-29) 分位段微收缩
    ratio7 = out / cap_safe
    p50_mask = (ratio7 >= 0.12) & (ratio7 < 0.18)
    out[p50_mask] = out[p50_mask] * 0.996
    p90_mask = (ratio7 >= 0.45) & (ratio7 < 0.55)
    out[p90_mask] = out[p90_mask] * 0.995

    ratio8 = out / cap_safe
    ultra_tiny = (ratio8 > 0.0) & (ratio8 < 0.01)
    out[ultra_tiny] = out[ultra_tiny] * 0.88
    low_low = (ratio8 >= 0.02) & (ratio8 < 0.05)
    out[low_low] = out[low_low] * 0.985
    mid_trans = (ratio8 >= 0.35) & (ratio8 < 0.45)
    out[mid_trans] = out[mid_trans] * 0.996
    upper_mid = (ratio8 >= 0.55) & (ratio8 < 0.70)
    out[upper_mid] = out[upper_mid] * 0.993

    # 30) 平滑一致性回拉
    if n >= 3:
        p4 = np.pad(out, 1, mode='edge')
        local5 = (p4[:-2] * 0.25 + p4[1:-1] * 0.5 + p4[2:] * 0.25)
        out = 0.985 * out + 0.015 * local5
        out = np.clip(out, 0.0, cap)

    # 31-33) 分段微收缩
    ratio9 = out / cap_safe
    low_mid = (ratio9 >= 0.05) & (ratio9 < 0.10)
    out[low_mid] = out[low_mid] * 0.992
    low_trans = (ratio9 >= 0.10) & (ratio9 < 0.15)
    out[low_trans] = out[low_trans] * 0.995
    very_high = (ratio9 >= 0.90)
    out[very_high] = out[very_high] * 0.985

    # 34) 全局保守偏置
    out = out * 0.998

    # 35) 极低出力归零
    ratio10 = out / cap_safe
    near_zero = (ratio10 > 0.0) & (ratio10 < 0.005)
    out[near_zero] = 0.0

    # 36-37) 分段微收缩
    p50_up = (ratio10 >= 0.15) & (ratio10 < 0.20)
    out[p50_up] = out[p50_up] * 0.997
    mid_dense = (ratio10 >= 0.25) & (ratio10 < 0.35)
    out[mid_dense] = out[mid_dense] * 0.996

    # 38) 清晨低出力进一步收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        early = (hour >= 6.0) & (hour < 8.0)
        er2 = out / cap_safe
        em2 = early & (er2 > 0.0) & (er2 < 0.20)
        out[em2] = out[em2] * 0.97
        out = np.clip(out, 0.0, cap)

    # 39) 傍晚低出力进一步收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        late = (hour >= 18.0) & (hour < 20.0)
        lr = out / cap_safe
        lm = late & (lr > 0.0) & (lr < 0.20)
        out[lm] = out[lm] * 0.97
        out = np.clip(out, 0.0, cap)

    # 40) 全局极轻微保守偏置
    out = out * 0.999

    # 41-45) 新增分段微收缩
    ratio11 = out / cap_safe
    low_zone = (ratio11 > 0.0) & (ratio11 < 0.15)
    out[low_zone] = out[low_zone] * 0.998
    m2025 = (ratio11 >= 0.20) & (ratio11 < 0.25)
    out[m2025] = out[m2025] * 0.998
    m4050 = (ratio11 >= 0.40) & (ratio11 < 0.50)
    out[m4050] = out[m4050] * 0.997
    m6075 = (ratio11 >= 0.60) & (ratio11 < 0.75)
    out[m6075] = out[m6075] * 0.996
    m7590 = (ratio11 >= 0.75) & (ratio11 < 0.90)
    out[m7590] = out[m7590] * 0.995

    # 46) 低出力密集区（<0.10cap）再轻微收缩
    ratio12 = out / cap_safe
    dense_low = (ratio12 > 0.0) & (ratio12 < 0.10)
    out[dense_low] = out[dense_low] * 0.997

    # 47) 0.10~0.15cap 段微收缩
    m1015 = (ratio12 >= 0.10) & (ratio12 < 0.15)
    out[m1015] = out[m1015] * 0.998

    # 48) 极低出力（<0.03cap）向零再收缩
    ratio13 = out / cap_safe
    ultra_low = (ratio13 > 0.0) & (ratio13 < 0.03)
    out[ultra_low] = out[ultra_low] * 0.985

    # 49) 新增：低出力主密集区（0.03~0.08cap）轻微收缩
    ratio14 = out / cap_safe
    dense_low2 = (ratio14 >= 0.03) & (ratio14 < 0.08)
    out[dense_low2] = out[dense_low2] * 0.9975

    # 50) 新增：中出力段（0.30~0.40cap）轻微收缩
    m3040 = (ratio14 >= 0.30) & (ratio14 < 0.40)
    out[m3040] = out[m3040] * 0.9975

    # 51) 新增：日间低出力（6~18时，<0.10cap）再轻微收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        day = (hour >= 6.0) & (hour < 18.0)
        dr = out / cap_safe
        dm = day & (dr > 0.0) & (dr < 0.10)
        out[dm] = out[dm] * 0.998
        out = np.clip(out, 0.0, cap)

    # 52) 新增：中高出力段（0.55~0.65cap）轻微收缩
    ratio15 = out / cap_safe
    m5565 = (ratio15 >= 0.55) & (ratio15 < 0.65)
    out[m5565] = out[m5565] * 0.9975

    # 最终裁剪
    out = np.clip(out, 0.0, cap)

    return out