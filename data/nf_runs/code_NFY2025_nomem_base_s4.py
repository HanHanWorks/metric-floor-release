import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    out = pp.copy()
    if n == 0:
        return out

    # 画像：均值比≈0.22，56%<0.2cap，33%<0.1cap，爬坡温和(p99≈0.048cap)，
    # 步自相关0.997，日自相关0.39。ACC 口径对超预测更敏感。
    # 历史稳定在 86.19~86.20，整体收缩已接近最优，继续微调收益很小。
    # 本轮：略微降低整体收缩强度（保留更多预测形态），同时加强对
    # 高功率段和低功率段的非对称处理——ACC 对超预测惩罚更重，
    # 故高功率端多收缩、低功率端少扰动。

    # 1) 轻度整体收缩（向中位数拉，降低超预测风险）
    p50 = np.median(out)
    out = p50 + 0.955 * (out - p50)

    # 2) 低功率段 5 点中值平滑（去噪，保留形态）
    if n >= 5:
        med = pp.copy()
        med[2:-2] = np.median(
            np.stack([pp[:-4], pp[1:-3], pp[2:-2], pp[3:-1], pp[4:]]), axis=0
        )
        low_mask = pp < 0.22 * cap
        out[low_mask] = 0.36 * pp[low_mask] + 0.64 * med[low_mask]

        # 2b) 极低值段(<0.08cap)更强平滑，减少噪声抖动
        very_low = pp < 0.08 * cap
        if np.any(very_low):
            out[very_low] = 0.26 * pp[very_low] + 0.74 * med[very_low]

    # 3) 抑制尖峰：超过 0.72cap 向 0.72cap 收缩（略加强，超预测代价高）
    upper = 0.72 * cap
    hi_mask = out > upper
    out[hi_mask] = upper + 0.44 * (out[hi_mask] - upper)

    # 4) 近零值极小抬升，避免完全为 0
    tiny = 0.0015 * cap
    zero_mask = out < tiny
    out[zero_mask] = tiny + 0.5 * out[zero_mask]

    # 5) 正午前后轻微平滑（利用 hour 信息）
    hour = view['hour'].values.astype(float)
    midday = (hour >= 10) & (hour <= 14)
    if np.any(midday) and n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out[midday] = 0.88 * out[midday] + 0.12 * sm[midday]

    # 6) 高频残差抑制：基于步自相关≈0.997，预测已很平滑，
    # 仅对相邻差过大的点做轻微回归，避免引入额外抖动
    if n >= 3:
        d = np.diff(out)
        thr = 0.06 * cap
        big = np.abs(d) > thr
        if np.any(big):
            idx = np.where(big)[0]
            for i in idx:
                if i + 1 < n:
                    mid = 0.5 * (out[i] + out[i + 1])
                    out[i] = 0.87 * out[i] + 0.13 * mid
                    out[i + 1] = 0.87 * out[i + 1] + 0.13 * mid

    # 7) 非负且不超过容量
    out = np.clip(out, 0.0, cap)

    # 8) 防止常数输出
    if np.std(out) < 1e-6 * cap:
        out = out + 1e-4 * cap * np.sin(np.arange(n) * 0.1)

    return out