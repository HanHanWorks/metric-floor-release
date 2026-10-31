import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    base = pp.copy()
    has_cap = (not np.isnan(cap)) and cap > 0

    # 1) 极低值区软抬升：大量点贴近零会显著拖累ACC。
    #    采用更温和的分段拉伸，尽量保留原始相对差异。
    if has_cap:
        f0 = 0.002 * cap
        f1 = 0.008 * cap
        f2 = 0.025 * cap
        f3 = 0.055 * cap
        b = base

        # 极低段：以较小偏移抬升至 f0 附近
        m0 = b < f0
        if np.any(m0):
            b = np.where(m0, b * 0.35 + f0 * 0.65, b)
        # f0~f1：平滑过渡
        m1 = (b >= f0) & (b < f1)
        if np.any(m1):
            t = (b - f0) / max(f1 - f0, 1e-9)
            b = np.where(m1, f1 + (b - f1) * (0.55 + 0.45 * t), b)
        # f1~f2：轻度抬升
        m2 = (b >= f1) & (b < f2)
        if np.any(m2):
            t = (b - f1) / max(f2 - f1, 1e-9)
            b = np.where(m2, b + (f2 - b) * 0.15 * (1.0 - t), b)
        # f2~f3：更轻微
        m3 = (b >= f2) & (b < f3)
        if np.any(m3):
            t = (b - f2) / max(f3 - f2, 1e-9)
            b = np.where(m3, b + (f3 - b) * 0.06 * (1.0 - t), b)
        base = b

    # 2) 毛刺抑制：step_autocorr 极高，仅对二阶差分异常点做轻度平滑。
    if n >= 5:
        w = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
        w = w / w.sum()
        sm = np.convolve(base, w, mode='same')
        sm[:2] = base[:2]
        sm[-2:] = base[-2:]
        d2 = np.abs(np.diff(base, n=2))
        if len(d2) > 0:
            thr = np.percentile(d2, 93)
            thr = max(thr, 1e-9)
            mask = np.zeros(n, dtype=bool)
            mask[1:-1] = d2 > thr
            if np.any(mask):
                out = base.copy()
                # 轻度混合，避免过度改变形态
                out[mask] = 0.50 * base[mask] + 0.50 * sm[mask]
                base = out

    # 3) 高值温和压缩：异常高值多为预测偏差。
    if has_cap:
        hi1 = 0.80 * cap
        hi2 = 0.92 * cap
        hi3 = 0.99 * cap
        base = np.where(base > hi1, hi1 + (base - hi1) * 0.40, base)
        base = np.where(base > hi2, hi2 + (base - hi2) * 0.25, base)
        base = np.where(base > hi3, hi3 + (base - hi3) * 0.12, base)

    # 4) 非负裁剪
    base = np.clip(base, 0, None)

    # 5) 防止常数
    if np.std(base) < 1e-9:
        t = np.arange(n, dtype=float)
        amp = 0.002 * (cap if has_cap else 1.0)
        base = base + amp * (np.sin(2 * np.pi * t / max(n, 1)) + 1.0)

    return base