import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    out = pp.copy()

    if n == 0:
        return out

    has_cap = np.isfinite(cap) and cap > 0
    if not has_cap:
        cap = 1.0

    # 0) 轻度平滑：只在明显非零段作用，保留爬坡起点
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.12 * out[:-2] + 0.76 * out[1:-1] + 0.12 * out[2:]
        mask = out > 0.04 * cap
        out = np.where(mask, sm, out)

    # 1) 极低出力段强收缩（系统性高估最严重区域）
    low = out < 0.04 * cap
    out = np.where(low, out * 0.22, out)

    # 2) 低出力段（4%-12%）下压
    low2 = (out >= 0.04 * cap) & (out < 0.12 * cap)
    out = np.where(low2, out * 0.42, out)

    # 3) 中低段（12%-30%）温和下压
    mid_low = (out >= 0.12 * cap) & (out < 0.30 * cap)
    out = np.where(mid_low, out * 0.66, out)

    # 4) 中段（30%-55%）轻微下压
    mid = (out >= 0.30 * cap) & (out < 0.55 * cap)
    out = np.where(mid, out * 0.82, out)

    # 5) 中高段（55%-70%）微幅下压
    mid_high = (out >= 0.55 * cap) & (out < 0.70 * cap)
    out = np.where(mid_high, out * 0.93, out)

    # 6) 高出力段（70%-85%）温和上抬补偿
    high = (out >= 0.70 * cap) & (out <= 0.85 * cap)
    out = np.where(high, out * 1.05, out)

    # 7) 接近容量上限处抑制过冲
    near_cap = out > 0.88 * cap
    out = np.where(near_cap, 0.88 * cap + (out - 0.88 * cap) * 0.15, out)

    # 8) 爬坡幅度限制（单遍轻量限幅）
    if n >= 2:
        max_ramp = 0.070 * cap
        prev = out[0]
        for i in range(1, n):
            delta = out[i] - prev
            if delta > max_ramp:
                out[i] = prev + max_ramp
            elif delta < -max_ramp:
                out[i] = prev - max_ramp
            prev = out[i]

    # 9) 物理约束
    out = np.clip(out, 0.0, cap)

    # 10) 防止常数序列
    if n > 0 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out