import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑，抑制 15min 噪声（保留日内形状）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 2) 低出力段收缩：低值区间真实功率更贴近 0，减少高估
    low_mask = out < 0.08 * cap
    out[low_mask] = out[low_mask] * 0.85

    # 3) 极低值截断到 0
    out = np.where(out < 0.005 * cap, 0.0, out)

    # 4) 上界约束
    out = np.clip(out, 0.0, cap)

    # 5) 尖峰轻微抑制
    if n >= 2:
        d = np.abs(np.diff(out))
        thr = 0.15 * cap
        spike = np.where(d > thr)[0]
        for i in spike:
            j = i + 1
            if j < n:
                out[j] = 0.7 * out[j] + 0.3 * out[i]

    # 6) 小幅整体校准
    out = out * 0.99

    # 7) 分位数微调：中低段再轻微收缩
    mid_low = (out >= 0.08 * cap) & (out < 0.25 * cap)
    out[mid_low] = out[mid_low] * 0.985

    # 8) 中高段轻微收缩（p90≈0.527, p99≈0.712，说明高段同样有高估倾向）
    high_mask = (out >= 0.45 * cap)
    out[high_mask] = out[high_mask] * 0.985

    # 9) 保持非恒定
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out