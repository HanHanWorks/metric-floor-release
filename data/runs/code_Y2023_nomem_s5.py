import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 历史 ACC 单调上升，低段收缩方向持续有效。本轮继续沿该方向推进，
    # 并对低段收缩与平滑做更细的分段处理以进一步降低小误差占比。
    # 新增：在收缩前先对极低段做一次噪声抑制平滑，避免尖峰被保留后放大。

    # 0) 极低段预平滑（仅在 <0.10 cap 区域），压制孤立尖峰
    if n >= 5:
        pre = out.copy()
        pre[2:-2] = (0.15 * out[:-4] + 0.20 * out[1:-3] + 0.30 * out[2:-2]
                     + 0.20 * out[3:-1] + 0.15 * out[4:])
        pre[0] = 0.60 * out[0] + 0.30 * out[1] + 0.10 * out[2]
        pre[1] = 0.30 * out[0] + 0.50 * out[1] + 0.20 * out[2]
        pre[-1] = 0.60 * out[-1] + 0.30 * out[-2] + 0.10 * out[-3]
        pre[-2] = 0.30 * out[-1] + 0.50 * out[-2] + 0.20 * out[-3]
        mask_pre = out < 0.10 * cap
        out = np.where(mask_pre, np.minimum(out, pre), out)

    # 1) 极低出力段向 0 收缩（噪声主导区）——继续轻微下压
    low = out < 0.06 * cap
    out = np.where(low, out * 0.632, out)

    # 1b) 次低段（0.06~0.08 cap）轻微收缩——继续轻微下压
    low2 = (out >= 0.06 * cap) & (out < 0.08 * cap)
    out = np.where(low2, out * 0.652, out)

    # 2) 低出力段 5 点加权平滑（仅低段），权重再平衡：中心再降，两侧再升
    if n >= 5:
        sm = out.copy()
        sm[2:-2] = (0.12 * out[:-4] + 0.22 * out[1:-3] + 0.32 * out[2:-2]
                    + 0.23 * out[3:-1] + 0.11 * out[4:])
        sm[0] = 0.55 * out[0] + 0.33 * out[1] + 0.12 * out[2]
        sm[1] = 0.28 * out[0] + 0.52 * out[1] + 0.20 * out[2]
        sm[-1] = 0.55 * out[-1] + 0.33 * out[-2] + 0.12 * out[-3]
        sm[-2] = 0.28 * out[-1] + 0.52 * out[-2] + 0.20 * out[-3]
        mask = out < 0.22 * cap
        out = np.where(mask, sm, out)

    # 3) 中等偏低段向 0 收缩（0.22~0.45 cap）——继续轻微下压
    mid_low = (out >= 0.22 * cap) & (out < 0.45 * cap)
    out = np.where(mid_low, out * 0.878, out)

    # 4) 中等段（0.45~0.65 cap）——继续轻微下压
    mid = (out >= 0.45 * cap) & (out < 0.65 * cap)
    out = np.where(mid, out * 0.924, out)

    # 5) 中高段（0.65~0.85 cap）——继续轻微下压
    mid_high = (out >= 0.65 * cap) & (out < 0.85 * cap)
    out = np.where(mid_high, out * 0.966, out)

    # 6) 高段（>=0.85 cap）——继续轻微下压
    high = out >= 0.85 * cap
    out = np.where(high, out * 0.982, out)

    # 7) 物理边界裁剪
    out = np.clip(out, 0.0, cap)

    # 8) 防止常数输出
    if np.allclose(out, out[0]):
        t = np.arange(n, dtype=float)
        out = out + 0.001 * cap * np.sin(2 * np.pi * t / max(n, 1))

    return out