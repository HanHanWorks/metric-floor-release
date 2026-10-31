import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 轻度平滑，抑制 15 分钟级噪声，保留日周期形态
    if n >= 3:
        w = 3
        pad = w // 2
        xp = np.pad(x, (pad, pad), mode='edge')
        kernel = np.ones(w) / w
        xs = np.convolve(xp, kernel, mode='valid')
        x = 0.75 * x + 0.25 * xs

    # 2) 基于 hour 的日周期校正：温和向小时中位数回归（中位数比均值更抗尖峰）
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        bins = np.floor(hour).astype(int) % 24
        day_med = np.zeros(24)
        day_cnt = np.zeros(24)
        for b in range(24):
            mask = bins == b
            c = int(mask.sum())
            day_cnt[b] = c
            if c > 0:
                day_med[b] = np.median(pp[mask])
        valid = day_cnt > 0
        if valid.any():
            # 缺失小时用整体中位数填充
            global_med = float(np.median(pp))
            day_med[~valid] = global_med
            ref = day_med[bins]
            resid = x - ref
            ratio = np.clip(pp / cap, 0.0, 1.0)
            # 低出力段收缩略强（减少虚高），高出力段收缩弱（保留形态）
            shrink = 0.08 + 0.10 * (1.0 - ratio)
            x = ref + resid * (1.0 - shrink)

    # 3) 物理约束
    x = np.clip(x, 0.0, cap)

    # 4) 防止常数输出
    if np.std(x) < 1e-6:
        if 'hour' in view.columns:
            hour = view['hour'].values.astype(float)
            x = x + 0.01 * cap * np.sin(2 * np.pi * hour / 24.0)
        else:
            x = x + 0.01 * cap * np.sin(np.linspace(0, 2 * np.pi, n))

    return x