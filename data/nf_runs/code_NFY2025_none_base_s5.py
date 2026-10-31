import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 去尖刺：仅修正明显离群点（保留形态）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        diff = out - sm
        thresh = 0.025 * cap
        mask = np.abs(diff) > thresh
        out[mask] = sm[mask]

    # 2) 非负裁剪与上限
    out = np.clip(out, 0.0, cap)

    # 3) 极低值温和抬升（欠预测敏感，但克制幅度以保 ACC）
    low_mask = out < 0.02 * cap
    if np.any(low_mask):
        out[low_mask] = out[low_mask] * 0.6 + 0.008 * cap

    # 4) 整体轻微上偏：欠预测惩罚更重，幅度保持克制
    ramp_p95 = 0.0178 * cap
    bias = 0.22 * ramp_p95
    out = out + bias

    # 5) 二次平滑：仅对小幅抖动平滑
    if n >= 5:
        sm2 = out.copy()
        sm2[2:-2] = (0.15 * out[:-4] + 0.2 * out[1:-3] + 0.3 * out[2:-2]
                     + 0.2 * out[3:-1] + 0.15 * out[4:])
        d2 = np.abs(sm2 - out)
        m2 = (d2 < 0.012 * cap)
        out[m2] = sm2[m2]

    out = np.clip(out, 0.0, cap)

    # 6) 针对高比例低出力场景做温和抬升
    mid_low = (out > 0.02 * cap) & (out < 0.35 * cap)
    if np.any(mid_low):
        out[mid_low] = out[mid_low] + 0.004 * cap

    # 7) 极低段相对抬升（克制）
    very_low = (out > 0.0) & (out < 0.15 * cap)
    if np.any(very_low):
        out[very_low] = out[very_low] * 1.015 + 0.002 * cap

    out = np.clip(out, 0.0, cap)

    # 8) 基于日自相关与高步自相关的时序一致性修正
    if n >= 7:
        med = out.copy()
        for i in range(2, n - 2):
            med[i] = np.median(out[i-2:i+3])
        d3 = out - med
        m3 = np.abs(d3) > 0.02 * cap
        out[m3] = 0.5 * out[m3] + 0.5 * med[m3]

    # 9) 针对极低出力做保底微抬
    tiny = (out >= 0.0) & (out < 0.05 * cap)
    if np.any(tiny):
        out[tiny] = out[tiny] + 0.003 * cap

    out = np.clip(out, 0.0, cap)

    # 10) 高步自相关(0.9977)下的轻度前向一致性稳定：
    #     以邻点均值做极小额约束，抑制单点孤立抖动
    if n >= 3:
        neighbor_avg = out.copy()
        neighbor_avg[1:-1] = 0.5 * out[:-2] + 0.5 * out[2:]
        d4 = np.abs(out - neighbor_avg)
        m4 = (d4 > 0.03 * cap)
        if np.any(m4):
            out[m4] = 0.6 * out[m4] + 0.4 * neighbor_avg[m4]

    # 11) 温和的欠预测补偿：对中低区间小幅加性抬升，兼顾 ACC 与偏差
    low_mid = (out > 0.05 * cap) & (out < 0.5 * cap)
    if np.any(low_mid):
        out[low_mid] = out[low_mid] + 0.0025 * cap

    out = np.clip(out, 0.0, cap)

    if len(out) != n:
        out = np.resize(out, n)

    return out