import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    n = len(pp)
    if n == 0:
        return pp
    cap = float(view['Cap'].iloc[0]) if n > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    out = pp.copy()

    # 数据画像：
    # - step_autocorr=0.9988 极高，序列极其平滑 → 平滑安全
    # - ramp_p95=0.0178 很小 → 无尖峰需保护
    # - frac_below_02cap=0.5783 → 多数时间低出力
    # - frac_zero=0.0015 → 几乎不到零
    # 历史稳定在 ACC~49.24，接近局部最优。
    # 本轮策略：加强低出力段的平滑抑制噪声（低段占比大，噪声对 ACC 影响大），
    # 同时温和扩展平滑带宽；高出力段几乎不动以保护峰值。

    # 1) 温和平滑：7 点高斯型核，比 5 点更柔和
    if n >= 7:
        w = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
        w = w / w.sum()
        padded = np.pad(out, (3, 3), mode='edge')
        smooth = np.convolve(padded, w, mode='valid')
        ratio = out / cap
        # 低出力段 alpha 上调到 0.6；0.5cap 以上快速衰减到 0，保护中高出力
        alpha = np.clip(1.0 - ratio / 0.5, 0.0, 1.0) * 0.6
        out = (1.0 - alpha) * out + alpha * smooth

    # 2) 极低值轻度收缩：仅对极小值做微小收缩，减少低段噪声
    tiny = out < 0.008 * cap
    out[tiny] = out[tiny] * 0.98

    # 3) 物理截断
    out = np.clip(out, 0.0, cap)

    # 4) 防止常数输出
    if n > 1 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out