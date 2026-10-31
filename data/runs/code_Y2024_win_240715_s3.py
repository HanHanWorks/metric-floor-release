import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 130.85
    hour = view['hour'].values.astype(float)
    n = len(pp)

    out = pp.copy()

    # 1) 轻度平滑：仅对明显毛刺做替换，保留真实爬坡
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        diff = np.abs(out - sm)
        thr = 0.02 * cap
        mask = diff > thr
        out = np.where(mask, sm, out)

    # 2) 低出力段收缩：日前预测在低出力时噪声大，向 0 收缩降低绝对误差
    low_mask = out < 0.05 * cap
    out = np.where(low_mask, out * 0.70, out)

    # 3) 夜间温和压低：夜间实际出力通常接近 0
    night = ((hour < 5.5) | (hour >= 22.0))
    small = out < 0.03 * cap
    out = np.where(night & small, out * 0.40, out)

    # 3b) 夜间中低出力进一步压低
    night_mid = night & (out >= 0.03 * cap) & (out < 0.10 * cap)
    out = np.where(night_mid, out * 0.62, out)

    # 3c) 夜间中出力段也适度压低
    night_hi = night & (out >= 0.10 * cap) & (out < 0.20 * cap)
    out = np.where(night_hi, out * 0.80, out)

    # 4) 中低出力段收缩（0.05~0.15cap），减少系统性高估
    mid_mask = (out >= 0.05 * cap) & (out < 0.15 * cap)
    out = np.where(mid_mask, out * 0.80, out)

    # 5) 中高出力段轻微收缩（0.15~0.35cap），抑制系统性高估
    midhi_mask = (out >= 0.15 * cap) & (out < 0.35 * cap)
    out = np.where(midhi_mask, out * 0.905, out)

    # 6) 高出力段轻微收缩（0.35~0.7cap），抑制尾部高估
    hi_mask = (out >= 0.35 * cap) & (out < 0.70 * cap)
    out = np.where(hi_mask, out * 0.950, out)

    # 7) 极高出力段几乎不动（>0.7cap），保留峰值信息
    vhi_mask = out >= 0.70 * cap
    out = np.where(vhi_mask, out * 0.985, out)

    # 8) 物理约束
    out = np.clip(out, 0.0, cap)

    # 9) 防止常数序列
    if np.allclose(out, out[0]):
        out = out + 1e-6 * cap * np.sin(np.arange(n) * 0.1)

    return out