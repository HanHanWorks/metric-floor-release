import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view else 0.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 中值滤波 M03（窗口半径 2 -> 5 点中值），抗孤立尖峰
    k = 2
    med = out.copy()
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        med[i] = np.median(pp[lo:hi])
    out = med

    # 2) 加权三点平滑 M02
    sm = out.copy()
    for i in range(1, n - 1):
        sm[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
    out = sm

    # 3) 温和爬坡限制（仅约束极端跳变）
    if cap > 0:
        max_step = 0.05 * cap
    else:
        max_step = 0.05 * max(np.max(out) if n else 0.0, 1e-6)
    for i in range(1, n):
        d = out[i] - out[i - 1]
        if d > max_step:
            out[i] = out[i - 1] + max_step
        elif d < -max_step:
            out[i] = out[i - 1] - max_step

    # 4) 容量约束
    if cap > 0:
        out = np.clip(out, 0.0, cap)
    else:
        out = np.clip(out, 0.0, None)

    # 5) 二次平滑
    sm2 = out.copy()
    for i in range(1, n - 1):
        sm2[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
    out = sm2

    # 6) 晚间/清晨低出力段轻微抬升（pp_mean_ratio=0.17，低出力为主）
    #    在出力极低段避免过度平滑导致的系统性低估
    if cap > 0:
        low_thresh = 0.05 * cap
        for i in range(n):
            if out[i] < low_thresh and pp[i] > out[i]:
                out[i] = 0.5 * out[i] + 0.5 * pp[i]

    # 7) 最终容量约束
    if cap > 0:
        out = np.clip(out, 0.0, cap)
    else:
        out = np.clip(out, 0.0, None)

    return out