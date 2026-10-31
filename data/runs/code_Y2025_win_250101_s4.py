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

    # 1) 轻度加权平滑（3点），抑制15分钟级噪声
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 2) 归一化到容量比例
    out = np.clip(out, 0.0, cap)
    ratio = out / cap

    # 3) 分段收缩映射（在第35/37轮基础上微调：低段略回吐、中高段略加深）
    adj = ratio.copy()

    # 极低段 (<0.05)：强收缩
    m = adj < 0.05
    adj[m] = adj[m] * 0.24

    # 低段 (0.05~0.18)
    m = (adj >= 0.05) & (adj < 0.18)
    adj[m] = adj[m] * 0.49

    # 中低段 (0.18~0.32)
    m = (adj >= 0.18) & (adj < 0.32)
    adj[m] = adj[m] * 0.67

    # 中段 (0.32~0.50)
    m = (adj >= 0.32) & (adj < 0.50)
    adj[m] = adj[m] * 0.795

    # 中高段 (0.50~0.68)
    m = (adj >= 0.50) & (adj < 0.68)
    adj[m] = adj[m] * 0.885

    # 高段 (0.68~0.85)
    m = (adj >= 0.68) & (adj <= 0.85)
    adj[m] = adj[m] * 0.955

    # 极高段 (>0.85)：保护峰值
    m = adj > 0.85
    adj[m] = adj[m] * 1.00

    out = adj * cap
    out = np.clip(out, 0.0, cap)

    # 4) 防止常数序列
    if np.allclose(out, out[0]):
        eps = 1e-6 * cap
        out = out + eps * np.sin(np.arange(n) * 0.1)
        out = np.clip(out, 0.0, cap)

    return out