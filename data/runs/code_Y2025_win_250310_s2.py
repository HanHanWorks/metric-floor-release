import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    if not np.isfinite(cap) or cap <= 0:
        mx = np.nanmax(pp) if np.isfinite(np.nanmax(pp)) else 0.0
        cap = mx if mx > 0 else 1.0

    n = len(pp)
    out = pp.copy()

    # 0) 先用时间/日周期做轻量的形态重建参考（对日内形态做稳健估计）
    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
    dow = view['dow'].values.astype(float) if 'dow' in view.columns else None

    # 1) 轻度平滑：短窗加权融合，抑制抖动同时保留日间形态
    if n >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        pad = 2
        padded = np.pad(pp, pad, mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        out = 0.7 * pp + 0.3 * smooth

    # 2) 物理边界裁剪
    out = np.clip(out, 0.0, cap)

    # 3) 低值区保护：避免平滑抬高夜间基线（更严格）
    low_mask = pp < 0.03 * cap
    out[low_mask] = np.minimum(out[low_mask], pp[low_mask] * 1.02 + 0.002 * cap)

    # 4) 分位收缩：对高值段温和收缩，防止尖峰过冲
    if n > 20:
        p98 = np.nanpercentile(pp, 98)
        p995 = np.nanpercentile(pp, 99.5)
        high_mask = pp > p98
        if np.any(high_mask):
            span = max(p995 - p98, 1e-6)
            excess = np.clip((pp[high_mask] - p98) / span, 0.0, 1.0)
            shrink = 0.93 - 0.12 * excess
            out[high_mask] = shrink * pp[high_mask] + (1.0 - shrink) * p98

    # 5) 基于日内形态的轻度纠偏：若同小时段历史形态较稳，向形态均值靠拢少量
    #    仅用当前 view 内的数据，按小时聚合得到该小时的稳健中位形态
    if hour is not None and n >= 96:
        try:
            hbins = np.floor(hour * 4).astype(int)  # 15min 段索引 0..95
            hbins = np.clip(hbins, 0, 95)
            # 用分钟粒度细化到 96 段
            minute_idx = np.arange(n) % 96
            med = np.zeros(96, dtype=float)
            cnt = np.zeros(96, dtype=float)
            for i in range(n):
                k = minute_idx[i]
                if np.isfinite(pp[i]):
                    med[k] += pp[i]
                    cnt[k] += 1.0
            valid = cnt > 0
            med[valid] /= cnt[valid]
            # 对缺失段用邻近插值
            if not np.all(valid):
                idx = np.arange(96)
                good = idx[valid]
                med = np.interp(idx, good, med[good])
            shape_ref = med[minute_idx]
            # 仅在形态参考合理时轻微融合
            out = 0.93 * out + 0.07 * shape_ref
        except Exception:
            pass

    # 6) 数值安全
    out = np.nan_to_num(out, nan=0.0, posinf=cap, neginf=0.0)
    out = np.clip(out, 0.0, cap)

    # 7) 兜底：避免常数序列
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out