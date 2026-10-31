import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑抑制 15min 噪声，但保留大爬坡
    if n >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        pad = 2
        padded = np.pad(pp, (pad, pad), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        ramp = np.abs(np.diff(pp, prepend=pp[0]))
        ramp_thr = 0.04 * cap
        w = np.clip(ramp / (ramp_thr + 1e-9), 0.0, 1.0)
        out = (1 - w) * smooth + w * pp

    # 2) 低功率段收缩
    low_mask = out < 0.05 * cap
    out = np.where(low_mask, out * 0.88, out)

    # 3) 中低功率段轻微收缩
    mid_low_mask = (out >= 0.05 * cap) & (out < 0.15 * cap)
    out = np.where(mid_low_mask, out * 0.95, out)

    # 4) 中功率段基本保持
    mid_mask = (out >= 0.15 * cap) & (out < 0.4 * cap)
    out = np.where(mid_mask, out * 0.985, out)

    # 5) 高功率段收缩
    mid_high_mask = (out >= 0.4 * cap) & (out <= 0.6 * cap)
    out = np.where(mid_high_mask, 0.4 * cap + (out - 0.4 * cap) * 0.9, out)
    high_mask = (out > 0.6 * cap) & (out <= 0.8 * cap)
    out = np.where(high_mask, 0.6 * cap + (out - 0.6 * cap) * 0.75, out)
    vhigh_mask = out > 0.8 * cap
    out = np.where(vhigh_mask, 0.8 * cap + (out - 0.8 * cap) * 0.45, out)

    # 6) 日间连续性约束：局部中位数回拉
    if n >= 9:
        pad_m = 4
        padded_out = np.pad(out, (pad_m, pad_m), mode='edge')
        med = np.array([np.median(padded_out[i:i+9]) for i in range(n)])
        dev = out - med
        dev_thr = 0.08 * cap
        back = np.clip((np.abs(dev) - dev_thr) / (dev_thr + 1e-9), 0.0, 1.0)
        out = out - np.sign(dev) * back * np.abs(dev) * 0.5

    # 7) 物理约束
    out = np.clip(out, 0.0, cap)

    # 8) 防止常数输出
    if np.std(out) < 1e-6:
        out = pp.copy()

    # 9) 极低功率段再收一档
    tiny_mask = out < 0.02 * cap
    out = np.where(tiny_mask, out * 0.7, out)

    # 11) 爬坡连续性修正：抑制孤立尖峰/深谷
    if n >= 3:
        prev = np.concatenate([out[:1], out[:-1]])
        nxt = np.concatenate([out[1:], out[-1:]])
        neigh_mean = 0.5 * (prev + nxt)
        dev2 = out - neigh_mean
        spike_thr = 0.06 * cap
        sp = np.clip((np.abs(dev2) - spike_thr) / (spike_thr + 1e-9), 0.0, 1.0)
        out = out - np.sign(dev2) * sp * np.abs(dev2) * 0.35

    # 12) 基于 hour 的夜间额外收缩
    hour = view['hour'].values.astype(float)
    night_mask = (hour < 6.0) | (hour >= 22.0)
    night_low = night_mask & (out < 0.25 * cap)
    out = np.where(night_low, out * 0.96, out)

    # 13) 中午时段轻微提升
    noon_mask = (hour >= 10.0) & (hour <= 14.0)
    noon_mid = noon_mask & (out >= 0.15 * cap) & (out < 0.5 * cap)
    out = np.where(noon_mid, out * 1.015, out)

    # 14) 分位数驱动的整体再校准
    mid_high2 = (out >= 0.45 * cap) & (out <= 0.7 * cap)
    out = np.where(mid_high2, out * 0.985, out)

    # 15) 尾部极高预测再压一档
    tail_high = out > 0.75 * cap
    out = np.where(tail_high, 0.75 * cap + (out - 0.75 * cap) * 0.7, out)

    # 16) 低功率段进一步向零收缩
    vlow = out < 0.10 * cap
    out = np.where(vlow, out * 0.92, out)

    # 17) 中低功率段轻微收缩
    lowmid = (out >= 0.10 * cap) & (out < 0.20 * cap)
    out = np.where(lowmid, out * 0.97, out)

    # 18) 基于 month 的季节性微调
    month = view['month'].values.astype(float)
    winter_mask = (month == 12) | (month == 1) | (month == 2)
    winter_mid = winter_mask & (out >= 0.15 * cap) & (out < 0.5 * cap)
    out = np.where(winter_mid, out * 0.985, out)

    # 19) 基于 ramp 幅值的局部去尖峰（仅对中等幅值过冲生效，保留真实大爬坡）
    if n >= 3:
        d = np.diff(out, prepend=out[0])
        d_next = np.concatenate([d[1:], d[-1:]])
        reversal = (d * d_next < 0)
        mag = np.minimum(np.abs(d), np.abs(d_next))
        rev_thr = 0.03 * cap
        rev_w = np.clip((mag - rev_thr) / (rev_thr + 1e-9), 0.0, 1.0) * reversal
        out = out - rev_w * (d_next * 0.4)

    # 20) 基于 median 的 p50 对齐（约 0.199cap），仅对偏高段轻压
    p50_target = 0.199 * cap
    above = out > p50_target
    excess = np.where(above, out - p50_target, 0.0)
    out = out - np.where(above, excess * 0.03, 0.0)

    # 21) 白天中高功率段轻微回收（预测整体偏高，考核对高估更敏感）
    day_mask = (hour >= 7.0) & (hour <= 19.0)
    day_midhigh = day_mask & (out >= 0.35 * cap) & (out < 0.65 * cap)
    out = np.where(day_midhigh, out * 0.99, out)

    # 22) 整体轻度向下偏置（应对 p50 偏低、长尾左侧更重的分布）
    out = out * 0.995

    # 23) 相邻点二阶差分平滑，仅对小幅抖动生效
    if n >= 3:
        d2 = np.diff(out, n=2, prepend=out[:1], append=out[-1:])
        wobble = np.abs(d2) < 0.015 * cap
        if np.any(wobble):
            sm = out.copy()
            sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
            out = np.where(wobble, sm, out)

    # 24) 新增：白天中段（0.2~0.35cap）进一步轻微下压，贴合 p50 偏低的分布
    day_lowmid = day_mask & (out >= 0.20 * cap) & (out < 0.35 * cap)
    out = np.where(day_lowmid, out * 0.992, out)

    # 25) 新增：清晨与傍晚过渡段的低功率再微收
    trans_mask = ((hour >= 6.0) & (hour < 8.0)) | ((hour >= 17.0) & (hour < 20.0))
    trans_low = trans_mask & (out < 0.20 * cap)
    out = np.where(trans_low, out * 0.98, out)

    # 26) 新增：高功率顶部（>0.65cap）再轻压，降低高估风险
    top_high = out > 0.65 * cap
    out = np.where(top_high, 0.65 * cap + (out - 0.65 * cap) * 0.85, out)

    out = np.clip(out, 0.0, cap)

    return out