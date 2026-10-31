import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float) if 'dow' in view.columns else np.zeros(len(pp))
    month = view['month'].values.astype(float) if 'month' in view.columns else np.ones(len(pp))
    n = len(pp)
    out = pp.copy()

    has_cap = np.isfinite(cap) and cap > 0
    base = cap if has_cap else (np.nanmax(pp) if n > 0 and np.isfinite(np.nanmax(pp)) else 1.0)
    if not np.isfinite(base) or base <= 0:
        base = 1.0

    # 1) 轻度平滑，抑制高频噪声但保留日周期
    if n >= 3:
        padded = np.pad(pp, (1, 1), mode='edge')
        smooth = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        out = 0.72 * pp + 0.28 * smooth

    # 2) 物理边界
    out = np.clip(out, 0.0, base)

    # 3) 夜间低出力收缩（分时段更细）
    night_deep = (hour < 4.5) | (hour >= 22.5)
    night_shallow = ((hour >= 4.5) & (hour < 6.0)) | ((hour >= 21.0) & (hour < 22.5))
    low_thr = 0.05 * base
    out = np.where(night_deep & (out < low_thr), out * 0.68, out)
    out = np.where(night_shallow & (out < low_thr), out * 0.80, out)

    # 4) 低出力分段收缩（画像：低出力区密集且偏高估）
    r = out / base
    very_low = (r >= 0.015) & (r < 0.08)
    out = np.where(very_low, out * 0.760, out)

    low_a = (r >= 0.08) & (r < 0.14)
    out = np.where(low_a, out * 0.808, out)

    low_mid = (r >= 0.14) & (r < 0.20)
    out = np.where(low_mid, out * 0.842, out)

    mid_low = (r >= 0.20) & (r < 0.30)
    out = np.where(mid_low, out * 0.888, out)

    # 5) 中高出力段轻微下调
    r = out / base
    mid_high = (r >= 0.30) & (r < 0.55)
    out = np.where(mid_high, out * 0.922, out)

    up_mid = (r >= 0.55) & (r <= 0.85)
    out = np.where(up_mid, out * 0.950, out)

    # 6) 高出力段保留峰值
    r = out / base
    high = r > 0.85
    out = np.where(high, np.minimum(out, base) * 0.988, out)

    # 7) 0.28~0.50 区间再压
    r = out / base
    band = (r >= 0.28) & (r < 0.50)
    out = np.where(band, out * 0.962, out)

    # 8) 保序微调（结合邻域均值与斜率）
    if n >= 3:
        padded = np.pad(out, (1, 1), mode='edge')
        neigh_mean = 0.5 * padded[:-2] + 0.5 * padded[2:]
        dev = out - neigh_mean
        thr = 0.058 * base
        pull = np.where(np.abs(dev) > thr, neigh_mean + 0.78 * dev, out)
        out = np.where(np.isfinite(pull), pull, out)

    # 9) 防止常数输出
    if np.std(out) < 1e-9:
        out = base * 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * hour / 24.0))

    # 最终边界
    out = np.clip(out, 0.0, base)
    return out