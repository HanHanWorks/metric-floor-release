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

    # 数据画像要点：
    # - pp_mean_ratio≈0.21, 62%<0.02cap, 39%<0.01cap -> 大量极低值区
    # - ramp 很小(p95≈0.019) -> 序列非常平滑，过度平滑会损失形态
    # - step_autocorr≈0.998 -> 短期强相关，日间相关弱(0.407)
    # - 考核 ACC 对超预测敏感，故整体保持轻微保守下调
    # 历史(86.64)已接近局部最优，本轮只做小幅精修：
    #   1) 平滑权重略微收敛，保留形态
    #   2) 高值段下调幅度回收一点点（此前 0.972 稍激进，回收到 0.976）
    #   3) 中值段维持温和下调
    #   4) 极低值区抬升幅度略增，稳定相对误差

    # 1) 极轻平滑
    if n >= 3:
        padded = np.pad(out, 1, mode='edge')
        smooth = 0.12 * padded[0:n] + 0.76 * padded[1:n+1] + 0.12 * padded[2:n+2]
        out = smooth

    # 2) 极低值区（<0.01cap，占39%）：轻微抬升
    very_low = pp < 0.01 * cap
    if very_low.any():
        out[very_low] = 0.88 * out[very_low] + 0.12 * (0.006 * cap)

    # 3) 低值区（0.01~0.05cap）：保持形态，极轻微下调
    low = (pp >= 0.01 * cap) & (pp < 0.05 * cap)
    if low.any():
        out[low] = out[low] * 0.993

    # 4) 中值区（0.05~0.5cap）：温和下调
    mid = (pp >= 0.05 * cap) & (pp < 0.5 * cap)
    if mid.any():
        out[mid] = out[mid] * 0.988

    # 5) 高值区（>=0.5cap）：下调幅度略回收
    high = pp >= 0.5 * cap
    if high.any():
        out[high] = out[high] * 0.976

    # 6) 抑制异常尖峰（>0.9cap 极少）
    spike = pp > 0.9 * cap
    if spike.any():
        out[spike] = 0.88 * out[spike] + 0.12 * (0.9 * cap)

    # 7) 非负且不超过容量
    out = np.clip(out, 0.0, cap)

    # 8) 防止常数输出
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out