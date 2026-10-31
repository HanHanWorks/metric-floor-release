import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    out = pp.copy()
    n = len(out)
    if n == 0:
        return out

    # 1) 保守的 3 点加权平滑（序列本身平滑，仅抑制小噪声）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        diff = np.abs(out - sm)
        mask = diff < 0.012 * cap
        out = np.where(mask, sm, out)

    # 2) 极低值软收缩：近零噪声区（<1% cap）向 0 靠拢
    low_thr = 0.01 * cap
    out = np.where(out < low_thr, out * 0.4, out)

    # 3) 低出力段（1%~5% cap）轻微收缩，抑制小波动
    mid_thr = 0.05 * cap
    out = np.where((out >= low_thr) & (out < mid_thr), out * 0.88, out)

    # 4) 中低出力段（5%~15% cap）非常轻微收缩，减少爬坡噪声
    mid2_thr = 0.15 * cap
    out = np.where((out >= mid_thr) & (out < mid2_thr), out * 0.97, out)

    # 5) 物理裁剪
    out = np.clip(out, 0.0, cap)

    # 6) 防止常数输出
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out