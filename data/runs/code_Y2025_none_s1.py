import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    n = len(pp)
    if n == 0:
        return pp
    if cap <= 0:
        cap = 1.0

    out = pp.copy()

    # 0) 高发段轻微下压，抑制尖峰高估（高段样本稀少，尖峰易拖累ACC）
    high_thr = 0.85 * cap
    high_mask = out > high_thr
    if np.any(high_mask):
        out[high_mask] = 0.96 * out[high_mask] + 0.04 * high_thr

    # 1) 极温和 3 点平滑：仅作用在非常平坦的段（避免抹掉真实爬坡）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        diff = np.abs(out[1:-1] - 0.5 * (out[:-2] + out[2:]))
        thr = 0.010 * cap
        mask = diff < thr
        out[1:-1] = np.where(mask, sm[1:-1], out[1:-1])

    # 2) 低出力段轻微上抬（针对低段系统性低估；但克制幅度，避免伤害MASE）
    low_thr = 0.02 * cap
    low_mask = out < low_thr
    if np.any(low_mask):
        idxs = np.where(low_mask)[0]
        ref = out.copy()
        for i in idxs:
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            local = out[lo:hi]
            local_hi = local[local >= low_thr]
            if len(local_hi) > 0:
                ref[i] = 0.80 * out[i] + 0.20 * np.mean(local_hi)
        out = ref

    # 3) 当天内按 4 小时分箱的轻收缩（稳定日内形态，轻微向箱均值靠拢）
    if 'hour' in view.columns and n >= 8:
        hour = view['hour'].values.astype(float)
        day_id = np.floor(np.arange(n) / 96.0).astype(int)
        out2 = out.copy()
        for d in np.unique(day_id):
            idx = np.where(day_id == d)[0]
            if len(idx) < 4:
                continue
            h = hour[idx]
            bins = np.floor(h / 4.0).astype(int)
            for b in np.unique(bins):
                bidx = idx[bins == b]
                if len(bidx) < 2:
                    continue
                m = np.mean(out[bidx])
                out2[bidx] = 0.93 * out[bidx] + 0.07 * m
        out = out2

    # 4) 温和抑制过大单步跳变（ramp_p99 约 0.03cap，限幅略高于它）
    if n >= 2:
        ramp_lim = 0.040 * cap
        for i in range(1, n):
            dd = out[i] - out[i - 1]
            if dd > ramp_lim:
                out[i] = out[i - 1] + ramp_lim
            elif dd < -ramp_lim:
                out[i] = out[i - 1] - ramp_lim

    # 5) 边界裁剪
    out = np.clip(out, 0.0, cap)

    # 6) 防止常数序列
    if n > 0 and np.allclose(out, out[0]):
        t = np.arange(n, dtype=float)
        out = out + 1e-3 * cap * np.sin(2 * np.pi * t / max(n, 1))

    return out