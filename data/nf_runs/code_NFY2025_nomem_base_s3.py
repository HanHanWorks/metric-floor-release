import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 低出力段自适应平滑：出力越低平滑越强（去噪），高出力段几乎不动（保留爬坡）
    if n >= 5:
        k = 5
        pad = k // 2
        padded = np.pad(pp, (pad, pad), mode='edge')
        kernel = np.ones(k) / k
        smooth = np.convolve(padded, kernel, mode='valid')
        if np.isfinite(cap) and cap > 0:
            ratio = np.clip(pp / cap, 0.0, 1.0)
            # 低出力区(0~0.3cap)平滑权重高，高出力区快速衰减
            w = np.clip(0.65 - 1.6 * ratio, 0.05, 0.65)
        else:
            w = np.full(n, 0.45)
        out = w * smooth + (1.0 - w) * pp

    # 2) 物理边界裁剪
    out = np.maximum(out, 0.0)
    if np.isfinite(cap) and cap > 0:
        out = np.minimum(out, cap)

    # 3) 极小值收缩：压低近零噪声，但保留真实小出力形态
    if np.isfinite(cap) and cap > 0:
        tiny = 0.006 * cap
        mask = out < tiny
        out[mask] = out[mask] * 0.75

    # 4) 防止常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0.0, 1e-6, n)

    return out