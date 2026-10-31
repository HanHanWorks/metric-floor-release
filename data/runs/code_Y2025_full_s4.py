import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0

    # 1) 中值滤波去孤立尖峰（M03）
    if n >= 5:
        med = np.empty(n)
        med[:2] = np.median(pp[:5])
        med[-2:] = np.median(pp[-5:])
        for i in range(2, n - 2):
            med[i] = np.median(pp[i - 2:i + 3])
        if has_cap:
            dev = np.abs(pp - med)
            thr = 0.05 * cap
            mask = dev > thr
            out[mask] = med[mask]
        else:
            out = med
    elif n >= 3:
        med = np.empty(n)
        med[0] = np.median(pp[:3])
        med[-1] = np.median(pp[-3:])
        for i in range(1, n - 1):
            med[i] = np.median(pp[i - 1:i + 2])
        out = med

    # 2) 轻度平滑（M02）：加权三点，低功率段减弱
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        sm[0] = 0.6 * out[0] + 0.4 * out[1]
        sm[-1] = 0.6 * out[-1] + 0.4 * out[-2]
        if has_cap:
            low_mask = out < 0.15 * cap
            sm[low_mask] = 0.3 * out[low_mask] + 0.7 * sm[low_mask]
        out = sm

    # 3) 容量约束
    if has_cap:
        out = np.clip(out, 0.0, cap)

    # 4) 爬坡限制
    if n >= 2 and has_cap:
        max_ramp = 0.06 * cap
        for i in range(1, n):
            d = out[i] - out[i - 1]
            if d > max_ramp:
                out[i] = out[i - 1] + max_ramp
            elif d < -max_ramp:
                out[i] = out[i - 1] - max_ramp
        out = np.clip(out, 0.0, cap)

    # 5) 日周期一致性校正
    if n >= 96 and has_cap:
        hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
        if hour is not None:
            hour_med = np.full(24, np.nan)
            for h in range(24):
                m = (hour >= h) & (hour < h + 1)
                if m.sum() > 0:
                    hour_med[h] = np.median(pp[m])
            if np.isnan(hour_med).any():
                valid = ~np.isnan(hour_med)
                if valid.sum() >= 2:
                    idx = np.arange(24)
                    hour_med = np.interp(idx, idx[valid], hour_med[valid])
            if not np.isnan(hour_med).any():
                h_idx = np.clip(np.floor(hour).astype(int), 0, 23)
                ref = np.clip(hour_med[h_idx], 0.0, cap)
                out = 0.85 * out + 0.15 * ref
                out = np.clip(out, 0.0, cap)

    # 6) 低出力段轻微收缩
    if has_cap:
        low = out < 0.10 * cap
        out[low] = out[low] * 0.92

    # 7) 极高出力段收缩
    if has_cap:
        high = out > 0.90 * cap
        out[high] = 0.97 * out[high] + 0.03 * (0.90 * cap)

    # 8) 短时平滑
    if n >= 5:
        sm2 = out.copy()
        sm2[2:-2] = 0.15 * out[0:-4] + 0.35 * out[1:-3] + 0.35 * out[3:-1] + 0.15 * out[4:]
        out = sm2

    # 9) 高段温和收缩
    if has_cap:
        upper = 0.75 * cap
        hi2 = out > upper
        out[hi2] = 0.90 * out[hi2] + 0.10 * upper

    # 10) 极小出力收缩
    if has_cap:
        tiny = out < 0.05 * cap
        out[tiny] = out[tiny] * 0.90

    # 11) 最终轻度二次平滑
    if n >= 7:
        sm3 = out.copy()
        sm3[3:-3] = (0.1 * out[0:-6] + 0.2 * out[1:-5] + 0.4 * out[3:-3]
                     + 0.2 * out[5:-1] + 0.1 * out[6:])
        out = sm3

    # 12) 新增：基于当日形态的自适应对齐——用日内分位参考做轻微回归
    #     捕捉整体出力水平，避免过度平滑压低峰值
    if n >= 32 and has_cap:
        # 用原始 pp 的日内分位数作为形态参考
        q_pp = np.percentile(pp, [10, 50, 90])
        q_out = np.percentile(out, [10, 50, 90])
        # 若输出分布整体偏低，做轻微向上补偿；偏高则向下
        if q_out[1] > 1e-6:
            scale = q_pp[1] / q_out[1]
            scale = np.clip(scale, 0.92, 1.08)
            out = out * scale
        if has_cap:
            out = np.clip(out, 0.0, cap)

    # 13) 新增：夜间段(极低出力)向0温和靠拢，减少低功率高估
    if has_cap:
        hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
        if hour is not None:
            night = (hour < 6) | (hour >= 21)
            if night.any():
                nv = out[night]
                nv = np.where(nv < 0.08 * cap, nv * 0.93, nv)
                out[night] = nv
        out = np.clip(out, 0.0, cap)

    if has_cap:
        out = np.clip(out, 0.0, cap)

    return out