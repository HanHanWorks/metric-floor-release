import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else 120.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 极轻平滑（保留原始形态，step_autocorr 极高）
    if n >= 5:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = 0.90 * out + 0.10 * sm

    # 2) 物理边界
    out = np.clip(out, 0.0, cap)

    # 3) 低值区极轻收缩（低值误差权重高，避免放大虚假小出力）
    low_mask = out < 0.015 * cap
    out[low_mask] = out[low_mask] * 0.97

    # 4) 极小值归零（真实零值稀少，抑制虚假小功率噪声）
    tiny = out < 0.002 * cap
    out[tiny] = 0.0

    # 5) 峰值区仅贴边界
    out = np.clip(out, 0.0, cap)

    # 6) 斜坡约束：仅对极端跳变做轻度压缩
    if n >= 3:
        thr = 0.066 * cap
        d = np.diff(out)
        big = np.abs(d) > thr
        if np.any(big):
            d2 = d.copy()
            d2[big] = np.sign(d[big]) * (thr + 0.7 * (np.abs(d[big]) - thr))
            out2 = np.empty_like(out)
            out2[0] = out[0]
            out2[1:] = out[0] + np.cumsum(d2)
            out = np.clip(out2, 0.0, cap)

    # 7) 轻度逐日形状校正：利用日自相关(0.41)，弱融合相邻日同相位
    if n >= 192:
        try:
            day = 96
            m = n // day
            if m >= 2:
                blocks = out[:m * day].reshape(m, day)
                fused = blocks.copy()
                if m >= 3:
                    fused[1:-1] = (0.88 * blocks[1:-1]
                                   + 0.06 * blocks[:-2]
                                   + 0.06 * blocks[2:])
                out_new = out.copy()
                out_new[:m * day] = fused.reshape(-1)
                out = 0.97 * out + 0.03 * out_new
                out = np.clip(out, 0.0, cap)
        except Exception:
            pass

    # 8) 低出力段轻微抬升，防止系统性低估
    band = (out >= 0.02 * cap) & (out <= 0.10 * cap)
    if np.any(band):
        out[band] = out[band] + 0.004 * cap

    # 9) 中高段轻微下压，抑制 PP 上偏（p99 仅 0.75，p90 仅 0.55，高点易高估）
    hi_band = out > 0.55 * cap
    if np.any(hi_band):
        out[hi_band] = out[hi_band] - 0.003 * cap

    out = np.clip(out, 0.0, cap)

    # 10) 保底：避免常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out