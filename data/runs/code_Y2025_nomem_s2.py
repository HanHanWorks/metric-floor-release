import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    out = pp.copy()

    # 日前预测本身已高度平滑 (step_autocorr≈0.997)，过度平滑会损失信息。
    # 采用更轻的 5 点加权平滑，仅用于抑制孤立尖刺，且仅在非零段应用。
    if n >= 5:
        sm = out.copy()
        sm[2:-2] = (0.10 * out[:-4] + 0.20 * out[1:-3] + 0.40 * out[2:-2]
                    + 0.20 * out[3:-1] + 0.10 * out[4:])
        mask = out > 1e-6
        out = np.where(mask, sm, out)

    # ACC 口径下低估惩罚更重（漏报代价高），对低功率段做小幅抬升。
    # frac_below_02cap≈0.51 占多数，抬升需保守；frac_below_01cap≈0.29 更需关注。
    # 极低段 (<0.01cap) 抬升稍多，因为此处误差对 ACC 影响最大。
    very_low = out < 0.01 * cap
    out = np.where(very_low, out * 1.03, out)

    low = (out >= 0.01 * cap) & (out < 0.02 * cap)
    out = np.where(low, out * 1.025, out)

    low2 = (out >= 0.02 * cap) & (out < 0.05 * cap)
    out = np.where(low2, out * 1.015, out)

    mid_low = (out >= 0.05 * cap) & (out < 0.10 * cap)
    out = np.where(mid_low, out * 1.008, out)

    # 中段 (0.10~0.30 cap) 为 pp_p50 所在区间，保持原值不动。
    # 中高段 (0.30~0.70 cap) 轻微压缩，防止过冲。
    mid_high = (out >= 0.30 * cap) & (out < 0.70 * cap)
    out = np.where(mid_high, out * 0.997, out)

    # 高功率段 (>=0.70 cap) 压缩稍多，frac_above_09cap≈0.0065 极少，
    # 但一旦过冲对 ACC 影响大。
    high = (out >= 0.70 * cap) & (out < 0.90 * cap)
    out = np.where(high, out * 0.993, out)

    very_high = out >= 0.90 * cap
    out = np.where(very_high, out * 0.988, out)

    # 裁剪
    out = np.clip(out, 0.0, cap)

    # 保证非常数
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out