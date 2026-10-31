import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float).copy()
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0

    # 1) 中值滤波去毛刺（M03）
    if n >= 5:
        med = out.copy()
        for i in range(2, n - 2):
            med[i] = np.median(out[i-2:i+3])
        if has_cap:
            thr = max(0.02 * cap, 1e-6)
            dev = np.abs(out - med)
            mask = dev > thr
            out[mask] = med[mask]

    # 2) 轻度平滑（M02），保留爬坡
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        ramp = np.abs(np.diff(out))
        ramp_thr = max(0.03 * cap, 1e-6) if has_cap else np.inf
        keep = np.ones(n, dtype=bool)
        big = ramp > ramp_thr
        keep[1:] |= big
        keep[:-1] |= big
        out = np.where(keep, out, sm)

    # 3) 容量约束
    if has_cap:
        out = np.clip(out, 0.0, cap)

    # 4) 滞后混合（M09）
    if n >= 2:
        mix = out.copy()
        mix[1:] = 0.85 * out[1:] + 0.15 * out[:-1]
        out = mix

    # 5) 分位拉伸（M10 轻量）
    if has_cap:
        r = out / cap
        mid = (r >= 0.15) & (r <= 0.65)
        out[mid] = out[mid] * 1.03
        hi = r > 0.65
        out[hi] = out[hi] * 0.985
        out = np.clip(out, 0.0, cap)

    # 6) 爬坡限制（M08 轻量）
    if has_cap and n >= 2:
        max_step = 0.08 * cap
        for i in range(1, n):
            d = out[i] - out[i-1]
            if d > max_step:
                out[i] = out[i-1] + max_step
            elif d < -max_step:
                out[i] = out[i-1] - max_step

    # 7) 抬底（M05 轻量）
    if has_cap:
        floor = 0.015 * cap
        low = out < floor
        out[low] = 0.5 * out[low] + 0.5 * floor

    # 8) 低出力收缩（M06 轻量）
    if has_cap:
        r = out / cap
        very_low = (r > 0.01) & (r < 0.06)
        out[very_low] = out[very_low] * 0.96

    # 9) 日历修正（M11 轻量）
    if 'dow' in view.columns and has_cap:
        dow = view['dow'].values
        weekend = dow >= 5
        out[weekend] = out[weekend] * 0.985
        out = np.clip(out, 0.0, cap)

    # 10) 夜间归零修正
    if has_cap and 'hour' in view.columns:
        hour = view['hour'].values
        night = (hour < 5.5) | (hour > 20.5)
        r = out / cap
        low_night = night & (r < 0.10)
        out[low_night] = out[low_night] * 0.85
        out = np.clip(out, 0.0, cap)

    # 11) 日尺度一致性微调
    if has_cap and n >= 96:
        r = out / cap
        day_mean = r.mean()
        pp_r = pp / cap
        pp_mean = pp_r.mean()
        if pp_mean > 1e-6:
            ratio = pp_mean / max(day_mean, 1e-6)
            ratio = float(np.clip(ratio, 0.97, 1.03))
            out = out * ratio
            out = np.clip(out, 0.0, cap)

    # 12) 低出力段渐进式压缩
    if has_cap:
        r = out / cap
        band = (r > 0.0) & (r < 0.05)
        scale = 0.85 + 0.15 * (r[band] / 0.05)
        out[band] = out[band] * scale
        out = np.clip(out, 0.0, cap)

    # 13) 新增：日内形状保形 + 峰谷对齐（提高 ACC 的关键——让峰段更接近原始 PP 的峰形）
    if has_cap and n >= 96:
        # 以 96 点为单位按天处理，仅当整除时；否则按整段处理
        seg = n // 96 if n >= 96 else 0
        if seg >= 1:
            for d in range(seg):
                s = d * 96
                e = s + 96
                pp_seg = pp[s:e]
                out_seg = out[s:e]
                pp_r = pp_seg / cap
                out_r = out_seg / cap
                # 峰段：原始 PP 偏高的位置，略微向原始 PP 靠拢，强化峰值命中
                hi_mask = pp_r >= 0.5
                if hi_mask.any():
                    # 峰段向原始 PP 靠拢 20%（保留形状差异但增强峰值对齐）
                    out_seg[hi_mask] = 0.8 * out_seg[hi_mask] + 0.2 * pp_seg[hi_mask]
                # 谷段：原始 PP 很低的位置，保持当前压低策略（不再抬升）
                lo_mask = pp_r < 0.08
                if lo_mask.any():
                    out_seg[lo_mask] = 0.92 * out_seg[lo_mask] + 0.08 * pp_seg[lo_mask]
                out[s:e] = np.clip(out_seg, 0.0, cap)

    # 14) 新增：相邻点差分一致性（抑制锯齿，提升平滑命中）
    if has_cap and n >= 3:
        r = out / cap
        # 仅在中等出力区间做弱平滑，避免影响启停沿
        mid_band = (r > 0.1) & (r < 0.6)
        if mid_band.sum() >= 3:
            sm = out.copy()
            sm[1:-1] = 0.2 * out[:-2] + 0.6 * out[1:-1] + 0.2 * out[2:]
            apply_mask = np.zeros(n, dtype=bool)
            apply_mask[1:-1] = mid_band[1:-1]
            out = np.where(apply_mask, sm, out)
            out = np.clip(out, 0.0, cap)

    return out