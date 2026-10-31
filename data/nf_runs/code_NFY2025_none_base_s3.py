import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 保留日周期结构的轻度平滑（仅在明显抖动处平滑，避免破坏爬坡）
    if n >= 5:
        kernel = np.array([0.25, 0.5, 0.25])
        padded = np.pad(pp, (1, 1), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        # 逐点判断局部抖动幅度：抖动小则多平滑，抖动大（爬坡）则少平滑
        diff = np.abs(pp - smooth)
        if np.isfinite(cap) and cap > 0:
            # 以容量的相对比例作为自适应阈值
            t_small = 0.01 * cap
            t_large = 0.05 * cap
        else:
            med = np.median(diff)
            t_small = 0.5 * med
            t_large = 2.0 * med
        w = np.clip((t_large - diff) / (t_large - t_small + 1e-9), 0.0, 1.0)
        # 抖动越小权重越大（最多平滑 0.5）
        alpha = 0.5 * w
        out = (1.0 - alpha) * pp + alpha * smooth

    # 2) 非负裁剪
    out = np.clip(out, 0.0, None)

    # 3) 容量上限裁剪
    if np.isfinite(cap) and cap > 0:
        out = np.clip(out, 0.0, cap)

    # 4) 极小值归零（低于阈值视为噪声/停机）
    if np.isfinite(cap) and cap > 0:
        thr = 0.004 * cap
        out = np.where(out < thr, 0.0, out)

    # 5) 保证非恒定序列
    if np.allclose(out, out[0]):
        out = pp.copy()
        out = np.clip(out, 0.0, None)
        if np.isfinite(cap) and cap > 0:
            out = np.clip(out, 0.0, cap)

    return out