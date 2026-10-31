import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    has_cap = (not np.isnan(cap)) and cap > 0
    scale = cap if has_cap else 1.0

    # 0) 轻度去毛刺，保留真实爬坡
    if n >= 5:
        med = pp.copy()
        med[1:-1] = np.median(np.stack([pp[:-2], pp[1:-1], pp[2:]], axis=0), axis=0)
        kernel = np.array([0.15, 0.7, 0.15])
        smooth = np.convolve(med, kernel, mode='same')
        smooth[0] = pp[0]
        smooth[-1] = pp[-1]
        diff = np.abs(pp - smooth)
        blend = np.where(diff > 0.05 * scale, 0.12, 0.42)
        out = (1 - blend) * pp + blend * smooth

    if has_cap:
        out = np.clip(out, 0.0, cap)

    # 1) 低值区收缩（进一步强化，压低低功率段占比）
    if has_cap:
        low_mask = out < 0.06 * cap
        out[low_mask] = out[low_mask] * 0.50
        very_low = out < 0.02 * cap
        out[very_low] = out[very_low] * 0.33
        mid_low = (out >= 0.06 * cap) & (out < 0.15 * cap)
        out[mid_low] = out[mid_low] * 0.68
        sub_mid = (out >= 0.15 * cap) & (out < 0.25 * cap)
        out[sub_mid] = out[sub_mid] * 0.845
        sub_mid2 = (out >= 0.25 * cap) & (out < 0.35 * cap)
        out[sub_mid2] = out[sub_mid2] * 0.915
        sub_mid3 = (out >= 0.35 * cap) & (out < 0.45 * cap)
        out[sub_mid3] = out[sub_mid3] * 0.955
        sub_mid4 = (out >= 0.45 * cap) & (out < 0.55 * cap)
        out[sub_mid4] = out[sub_mid4] * 0.980

    # 2) 高值区温和收缩，保留高值信息
    if has_cap:
        high_mask = out > 0.92 * cap
        out[high_mask] = 0.92 * cap + (out[high_mask] - 0.92 * cap) * 0.66
        vhigh = (out > 0.85 * cap) & (out <= 0.92 * cap)
        out[vhigh] = out[vhigh] * 0.955

    # 3) 爬坡软限幅（略收紧）
    if has_cap and n > 1:
        max_ramp = 0.090 * cap
        diffs = np.diff(out)
        pos_excess = diffs - max_ramp
        pos_excess[pos_excess < 0] = 0
        neg_excess = -diffs - max_ramp
        neg_excess[neg_excess < 0] = 0
        adj = np.zeros(n)
        adj[1:] = -pos_excess + neg_excess
        out = out + adj
        out = np.clip(out, 0.0, cap)

    # 4) 日周期一致性
    if n >= 96 and has_cap:
        day = 96
        n_days = n // day
        if n_days >= 2:
            for d in range(1, n_days):
                s = d * day
                e = min(s + day, n)
                if e - s < day:
                    break
                prev_s = s - day
                cur = out[s:e].copy()
                prev = out[prev_s:prev_s + day]
                diff_day = np.abs(cur - prev)
                mask = diff_day < 0.08 * scale
                if np.any(mask):
                    cur[mask] = 0.72 * cur[mask] + 0.28 * prev[mask]
                    out[s:e] = cur
            for d in range(n_days - 2, -1, -1):
                s = d * day
                e = s + day
                nxt_s = s + day
                if nxt_s + day > n:
                    continue
                cur = out[s:e].copy()
                nxt = out[nxt_s:nxt_s + day]
                diff_day = np.abs(cur - nxt)
                mask = diff_day < 0.08 * scale
                if np.any(mask):
                    cur[mask] = 0.78 * cur[mask] + 0.22 * nxt[mask]
                    out[s:e] = cur

    # 5) 时段画像校准（继续压夜间/清晨，抬午间）
    if has_cap and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night = (hour < 6.0) | (hour >= 22.0)
        early_morning = (hour >= 6.0) & (hour < 8.0)
        late_evening = (hour >= 20.0) & (hour < 22.0)
        midday = (hour >= 10.0) & (hour <= 14.0)
        out[night] = out[night] * 0.675
        out[early_morning] = out[early_morning] * 0.795
        out[late_evening] = out[late_evening] * 0.845
        out[midday] = out[midday] * 1.045
        out = np.clip(out, 0.0, cap)

    # 6) 全局轻量整体收缩
    if has_cap:
        overall = out > 0.03 * cap
        out[overall] = out[overall] * 0.980

    if has_cap:
        out = np.clip(out, 0.0, cap)

    return out