import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    # 1) 中值滤波去孤立尖峰（M03）
    out = pp.copy()
    k = 2  # 窗口半径
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        med = np.median(pp[lo:hi])
        # 仅当偏离中位数过大时替换（M04 思想，阈值放宽避免过度修改）
        if cap > 0 and abs(pp[i] - med) > 0.15 * cap:
            out[i] = med
        else:
            out[i] = pp[i]

    # 2) 轻度平滑（M02）：加权三点，保留端点
    smoothed = out.copy()
    w = np.array([0.25, 0.5, 0.25])
    for i in range(1, n - 1):
        smoothed[i] = w[0] * out[i - 1] + w[1] * out[i] + w[2] * out[i + 1]
    out = smoothed

    # 3) 容量约束（M07）
    if cap > 0 and not np.isnan(cap):
        out = np.clip(out, 0.0, cap)

    # 4) 爬坡限制（M08）：相邻点变化幅度受限
    if cap > 0 and not np.isnan(cap):
        max_ramp = 0.20 * cap
        for i in range(1, n):
            diff = out[i] - out[i - 1]
            if diff > max_ramp:
                out[i] = out[i - 1] + max_ramp
            elif diff < -max_ramp:
                out[i] = out[i - 1] - max_ramp
        out = np.clip(out, 0.0, cap)

    return out