import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    out = pp.copy()
    if n == 0:
        return out

    # 回顾：第53轮 ACC=53.04，之后尝试加速超发保护/平滑/中段调整都没提升，
    # 甚至略降。说明第53轮的"低功率段收缩 + 上限保护 + 极轻平滑"结构已经接近局部最优。
    # 本轮回归第53轮核心结构，并做两处保守微调：
    #   (a) 极低功率段收缩从0.68改为0.70，避免过度压扁（MASE 惩罚）；
    #   (b) 接近满发保护从0.96cap放宽到0.97cap，减少对高功率段的过度压制，
    #       因为 frac_above_09cap≈0.0123 本身很少，过强收缩反而拉大偏差。
    # 其余段落与第53轮一致。

    eps = 0.01 * cap

    # 极低功率段（<0.01cap）：相对误差最敏感，收缩。
    small = pp < eps
    out[small] = pp[small] * 0.70

    # 低功率段（0.01cap ~ 0.08cap）：向该段中位数轻收缩。
    low = (pp >= eps) & (pp < 0.08 * cap)
    if np.any(low):
        anchor_low = np.median(pp[low])
        out[low] = pp[low] * 0.88 + anchor_low * 0.12

    # 低-中功率段（0.08cap ~ 0.20cap）：轻度收缩。
    low_mid = (pp >= 0.08 * cap) & (pp < 0.20 * cap)
    if np.any(low_mid):
        anchor_lm = np.median(pp[low_mid])
        out[low_mid] = pp[low_mid] * 0.95 + anchor_lm * 0.05

    # 中功率段（0.20cap ~ 0.60cap）：基本保持。
    mid = (pp >= 0.20 * cap) & (pp < 0.60 * cap)
    if np.any(mid):
        out[mid] = pp[mid] * 0.985

    # 中-高功率段（0.60cap ~ 0.95cap）：略降。
    mid_hi = (pp >= 0.60 * cap) & (pp <= 0.95 * cap)
    if np.any(mid_hi):
        out[mid_hi] = pp[mid_hi] * 0.985

    # 接近满发：上限保护。
    hi = pp > 0.95 * cap
    out[hi] = np.minimum(pp[hi], 0.97 * cap)

    # 极轻平滑：仅抑制毛刺，避开极低与极高端。
    if n >= 5:
        kernel = np.array([0.05, 0.10, 0.70, 0.10, 0.05])
        padded = np.concatenate(([out[0]] * 2, out, [out[-1]] * 2))
        smoothed = np.convolve(padded, kernel, mode='valid')
        mask = (pp >= 0.02 * cap) & (pp <= 0.95 * cap)
        out[mask] = smoothed[mask]

    # 数值安全
    out = np.clip(out, 0.0, cap)
    out = np.where(np.isfinite(out), out, pp)

    return out