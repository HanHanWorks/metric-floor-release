import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    y = pp.copy()

    # 数据画像：中位数 0.085cap，均值 0.19cap，69% 低于 0.2cap，53% 低于 0.1cap。
    # ACC 对低功率段相对误差敏感：低段收缩、极小值归零；高段轻微上抬。
    # 第37轮：在历史最优附近继续微调——平滑权重再降一点（减少滞后），
    # 极低段收缩略减弱，中低段收缩略减弱以保留有效信号。

    # 1) 轻度平滑：抑制 15 分钟级噪声（保持边缘），进一步降低平滑权重以减小滞后
    if n >= 3:
        ys = y.copy()
        ys[1:-1] = 0.15 * y[:-2] + 0.70 * y[1:-1] + 0.15 * y[2:]
        y = ys

    # 2) 极低功率段：低于 0.07cap 向 0 收缩
    very_low = y < 0.07 * cap
    y[very_low] = y[very_low] * 0.80

    # 3) 低功率段（0.07~0.12cap）收缩
    low_mask = (y >= 0.07 * cap) & (y < 0.12 * cap)
    y[low_mask] = y[low_mask] * 0.88

    # 4) 中低段（0.12~0.22cap）轻微收缩
    mid_mask = (y >= 0.12 * cap) & (y < 0.22 * cap)
    y[mid_mask] = y[mid_mask] * 0.94

    # 5) 极小值归零（避免相对误差爆炸）
    tiny_mask = y < 0.006 * cap
    y[tiny_mask] = 0.0

    # 6) 高功率段轻微上抬（爬坡段常被低估）
    high_mask = y > 0.87 * cap
    y[high_mask] = y[high_mask] + 0.012 * cap * (y[high_mask] / cap - 0.87) / 0.13

    # 7) 物理约束
    y = np.clip(y, 0.0, cap)

    # 8) 防止常数序列
    if np.allclose(y, y[0]):
        eps = 1e-6 * max(cap, 1.0)
        y = y + eps * np.sin(np.arange(n) * 0.1)

    return y