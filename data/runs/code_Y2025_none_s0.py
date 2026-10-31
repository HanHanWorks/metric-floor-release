import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp.copy()

    out = pp.copy()

    # 第58轮 ACC=55.92 为当前最优。本轮在该结构上继续微调：
    # 1) 进一步降低平滑中心权重（0.36->0.34），提升对 ramp 的响应；
    # 2) 低段收缩略加深，尤其 <0.03cap 与 0.03~0.08cap；
    # 3) 中高段上抬进一步收敛，避免过抬抖动；
    # 4) 0.60~0.75cap 段补偿继续保留但幅度略减。

    # 1) 轻平滑（7点三角权重，端点保持）
    if n >= 7:
        sm = out.copy()
        sm[3:-3] = (0.05 * out[:-6] + 0.09 * out[1:-5] + 0.15 * out[2:-4]
                    + 0.34 * out[3:-3]
                    + 0.15 * out[4:-2] + 0.09 * out[5:-1] + 0.05 * out[6:])
        sm[0] = out[0]
        sm[1] = out[1]
        sm[2] = out[2]
        sm[-3] = out[-3]
        sm[-2] = out[-2]
        sm[-1] = out[-1]
    elif n >= 5:
        sm = out.copy()
        sm[2:-2] = (0.1 * out[:-4] + 0.2 * out[1:-3] + 0.4 * out[2:-2]
                    + 0.2 * out[3:-1] + 0.1 * out[4:])
        sm[0] = out[0]
        sm[1] = out[1]
        sm[-2] = out[-2]
        sm[-1] = out[-1]
    elif n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        sm[0] = out[0]
        sm[-1] = out[-1]
    else:
        sm = out.copy()

    # 2) 低功率段细分收缩（略加深）
    m0 = sm < 0.03 * cap
    sm[m0] = sm[m0] * 0.925

    m1 = (sm >= 0.03 * cap) & (sm < 0.08 * cap)
    sm[m1] = sm[m1] * 0.948

    m2 = (sm >= 0.08 * cap) & (sm < 0.15 * cap)
    sm[m2] = sm[m2] * 0.966

    m3 = (sm >= 0.15 * cap) & (sm < 0.22 * cap)
    sm[m3] = sm[m3] * 0.980

    m4 = (sm >= 0.22 * cap) & (sm < 0.30 * cap)
    sm[m4] = sm[m4] * 0.990

    m5 = (sm >= 0.30 * cap) & (sm < 0.40 * cap)
    sm[m5] = sm[m5] * 0.996

    # 3) 中段（0.40~0.60cap）几乎不动
    m6 = (sm >= 0.40 * cap) & (sm < 0.60 * cap)
    sm[m6] = sm[m6] * 0.999

    # 4) 高功率段上抬幅度收敛
    m7 = (sm >= 0.60 * cap) & (sm < 0.75 * cap)
    sm[m7] = sm[m7] * 1.004

    m8 = sm >= 0.75 * cap
    sm[m8] = sm[m8] * 1.007

    # 5) 裁剪
    sm = np.clip(sm, 0.0, cap)

    # 6) 防常数
    if np.std(sm) < 1e-6:
        t = np.arange(n, dtype=float)
        sm = sm + 1e-3 * cap * np.sin(2 * np.pi * t / max(n, 1))

    return sm