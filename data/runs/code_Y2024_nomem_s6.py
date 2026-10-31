import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 80.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 数据 step_autocorr=0.9978 极高，序列本身非常平滑。
    # 上一版 5 点平滑已在第 48-53 轮稳定在 50.15-50.18，
    # 说明微调平滑权重空间已很小。这里保留极轻平滑但略微收窄，
    # 减少对已平滑信号二次引入的滞后。
    if n >= 5:
        sm = out.copy()
        sm[2:-2] = (0.12 * out[:-4] + 0.18 * out[1:-3] + 0.40 * out[2:-2]
                    + 0.18 * out[3:-1] + 0.12 * out[4:])
        out = sm

    # 物理约束：先裁到 [0, cap]
    out = np.clip(out, 0.0, cap)

    # 低出力段：frac_below_02cap=0.5075, frac_zero=0.0075。
    # 真实值多为小正值，轻微抬升以减小 MAE/MASE 在零附近的惩罚。
    # 上一版用 0.004*cap，这里略微下调阈值带宽，只抬极低段，避免污染中段。
    low_mask = out < 0.012 * cap
    out[low_mask] = np.maximum(out[low_mask], 0.0035 * cap)

    # 爬坡约束：ramp_p99=0.0572*cap, ramp_max=0.3811*cap。
    # 上一版 0.09*cap 较宽松，可能放过部分尖峰。
    # 收紧到 0.075*cap，更贴近 p99 与 max 之间，抑制极端尖峰而不伤正常波动。
    max_ramp = 0.075 * cap
    for i in range(1, n):
        diff = out[i] - out[i - 1]
        if diff > max_ramp:
            out[i] = out[i - 1] + max_ramp
        elif diff < -max_ramp:
            out[i] = out[i - 1] - max_ramp

    # 再次裁剪
    out = np.clip(out, 0.0, cap)

    # 防止常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out