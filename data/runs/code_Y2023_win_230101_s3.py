import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑：抑制高频抖动，保留爬坡结构
    if n >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        padded = np.pad(pp, (1, 1), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        out = 0.7 * pp + 0.3 * smooth

    # 2) 非负裁剪
    out = np.clip(out, 0.0, None)

    # 3) 容量上限裁剪
    if np.isfinite(cap) and cap > 0:
        out = np.clip(out, 0.0, cap)

    # 4) 极低值区间轻微平滑（抑制夜间噪声抖动）
    if np.isfinite(cap) and cap > 0:
        low_mask = (out > 0) & (out < 0.02 * cap)
        if np.any(low_mask):
            padded = np.pad(out, (1, 1), mode='edge')
            local_mean = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
            out[low_mask] = 0.5 * out[low_mask] + 0.5 * local_mean[low_mask]
        very_low = out < 0.005 * cap
        out[very_low] = out[very_low] * 0.5

    # 5) 防止常数输出
    if np.std(out) < 1e-9:
        return pp

    # 6) 基于日内形状的偏差校正（保守，弱化以避免破坏原有精度）
    if 'hour' in view.columns and np.isfinite(cap) and cap > 0:
        hour = view['hour'].values.astype(float)
        shape = np.exp(-0.5 * ((hour - 12.5) / 3.5) ** 2)
        day_mask = (hour >= 6.0) & (hour <= 19.0)
        if np.any(day_mask):
            day_out = out[day_mask]
            day_shape = shape[day_mask]
            if np.sum(day_shape) > 1e-9:
                shape_mean = np.sum(day_out * day_shape) / np.sum(day_shape)
                target = shape_mean * day_shape
                delta = 0.06 * (target - day_out)
                max_delta = 0.15 * np.abs(day_out) + 1e-6
                delta = np.clip(delta, -max_delta, max_delta)
                out[day_mask] = day_out + delta

    # 7) 爬坡限幅：抑制异常尖峰（ramp_p99=0.04cap，ramp_max=0.28cap）
    if np.isfinite(cap) and cap > 0 and n >= 2:
        ramp_limit = 0.06 * cap
        for i in range(1, n):
            d = out[i] - out[i - 1]
            if d > ramp_limit:
                out[i] = out[i - 1] + ramp_limit + 0.5 * (d - ramp_limit)
            elif d < -ramp_limit:
                out[i] = out[i - 1] - ramp_limit + 0.5 * (d + ramp_limit)

    # 8) 夜间压低（光伏夜间应接近0，抑制夜间虚高，弱化力度）
    if 'hour' in view.columns and np.isfinite(cap) and cap > 0:
        hour = view['hour'].values.astype(float)
        night_mask = (hour < 5.5) | (hour > 20.5)
        if np.any(night_mask):
            out[night_mask] = out[night_mask] * 0.85

    # 9) 再次裁剪
    out = np.clip(out, 0.0, None)
    if np.isfinite(cap) and cap > 0:
        out = np.clip(out, 0.0, cap)

    # 10) 防止常数输出
    if np.std(out) < 1e-9:
        return pp

    return out