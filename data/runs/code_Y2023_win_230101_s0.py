import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 1.0
    hour = view['hour'].values.astype(float)

    n = len(pp)
    if n == 0:
        return pp
    if cap <= 0:
        cap = 1.0

    # 1) 轻度平滑：抑制高频噪声，保留日周期结构
    if n >= 3:
        sm = pp.copy()
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        out = 0.6 * pp + 0.4 * sm
    else:
        out = pp.copy()

    # 2) 温和的分时段校正（预测侧统计驱动）
    h = hour
    day_mask = (h >= 6) & (h <= 18)
    night_mask = ~day_mask

    scale = np.ones(n)
    scale[night_mask] = 0.985
    scale[day_mask] = 1.008
    out = out * scale

    # 3) 爬坡平滑：限制相邻点突变（ramp_p99≈0.0335cap）
    max_step = 0.033 * cap
    for i in range(1, n):
        d = out[i] - out[i - 1]
        if d > max_step:
            out[i] = out[i - 1] + max_step
        elif d < -max_step:
            out[i] = out[i - 1] - max_step

    # 4) 物理边界裁剪
    out = np.clip(out, 0.0, cap)

    # 5) 保底：避免常数输出
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out