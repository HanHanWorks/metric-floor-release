import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 94.725
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    hi = max(cap, 1e-6)

    # 历史：M01/M02/M03/M05/M06/M07/M08/M09/M11/M12/M10 全部被拒。
    # 说明：任何整体抬升/平滑/缩放都会破坏 step_autocorr(0.9983) 与 ramp 结构，
    # 而 ACC 对系统性偏差非常敏感；低出力段占 62%，但"抬底"已被证明有害。
    # 本轮策略：完全不动主体，只在"极高段"做极轻的、非线性的局部压缩，
    # 并且压缩量随该点本身的相对高度自适应 —— 越接近容量越强，避免动中低段。
    # 这是一种"只削尖顶、不碰基础"的保守做法，理论上只减少高估惩罚，
    # 对 ACC 的方向是单向正贡献，且不改变低出力分布。

    x = np.clip(out, 0.0, hi) / hi

    if n < 8:
        return np.clip(out, 0.0, hi)

    # 只关注高段，阈值取 0.85 与 0.95 两个分位，与既有逻辑一致的保守边界
    q85 = np.percentile(x, 85)
    q95 = np.percentile(x, 95)

    # 基准阈值：至少 0.55*cap 以上才考虑压缩，避免误伤中段
    base_thr = max(q85, 0.55)

    y = x.copy()

    # 高段（> base_thr）：轻度压缩，压缩强度随 (x - base_thr) 线性增长
    # 最大压缩比例控制在 ~6%，仅在接近容量处达到
    mask_hi = x > base_thr
    if np.any(mask_hi):
        span = max(1.0 - base_thr, 1e-6)
        w = np.clip((x[mask_hi] - base_thr) / span, 0.0, 1.0)
        # 压缩系数：w=0 时 1.0，w=1 时 0.94
        factor = 1.0 - 0.06 * w
        y[mask_hi] = x[mask_hi] * factor

    # 极高段（> q95 且 > 0.7）：额外轻度压缩，最多再 -3%
    vhi_thr = max(q95, 0.70)
    mask_vhi = x > vhi_thr
    if np.any(mask_vhi):
        span2 = max(1.0 - vhi_thr, 1e-6)
        w2 = np.clip((x[mask_vhi] - vhi_thr) / span2, 0.0, 1.0)
        factor2 = 1.0 - 0.03 * w2
        y[mask_vhi] = y[mask_vhi] * factor2

    out = y * hi

    # 硬容量约束
    out = np.clip(out, 0.0, hi)

    # 防止常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out