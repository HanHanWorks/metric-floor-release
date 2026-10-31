import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 95.0
    hour = view['hour'].values.astype(float)
    n = len(pp)

    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑：抑制单点噪声（保留原始为主）
    med = pp.copy()
    if n >= 3:
        med[1:-1] = np.median(np.stack([pp[:-2], pp[1:-1], pp[2:]], axis=0), axis=0)
    out = 0.82 * pp + 0.18 * med

    # 2) 夜间收缩：22-6 点 PP 常高估，向 0 收缩
    h = np.mod(hour, 24.0)
    night = ((h >= 22) | (h < 6)).astype(float)
    out = out * (1.0 - 0.08 * night)

    # 3) 极低出力段向 0 收缩（相对误差大）
    low_mask = pp < 0.05 * cap
    out = np.where(low_mask, out * 0.95, out)

    # 4) 高出力段轻微下压（避免越限）
    high_mask = pp > 0.85 * cap
    out = np.where(high_mask, out * 0.985, out)

    # 5) 爬坡限幅（单步不超过 ~p99 ramp）
    max_ramp = 0.032 * cap
    if n >= 2:
        for i in range(1, n):
            diff = out[i] - out[i - 1]
            if diff > max_ramp:
                out[i] = out[i - 1] + max_ramp
            elif diff < -max_ramp:
                out[i] = out[i - 1] - max_ramp

    # 6) 非负裁剪
    out = np.clip(out, 0.0, cap)

    # 7) 低出力段向局部均值轻微靠拢（提升稳定性）
    if n >= 4:
        win = 5
        pad = win // 2
        padded = np.pad(out, pad, mode='edge')
        local_mean = np.convolve(padded, np.ones(win)/win, mode='valid')
        low2 = pp < 0.15 * cap
        out = np.where(low2, 0.90 * out + 0.10 * local_mean, out)

    # 8) 极低值保护：避免过度收缩到 0 导致 MASE 恶化
    tiny = (pp > 0.005 * cap) & (out < 0.3 * pp)
    out = np.where(tiny, 0.75 * pp + 0.25 * out, out)

    # 9) 中低出力段轻微抬升（该段占比大，ACC 口径下适度抬升可减少低估惩罚）
    mid_low = (pp >= 0.05 * cap) & (pp < 0.20 * cap)
    out = np.where(mid_low, out * 1.025, out)

    # 10) 低出力段轻微抬升（进一步减少低估惩罚）
    very_low = (pp >= 0.02 * cap) & (pp < 0.05 * cap)
    out = np.where(very_low, out * 1.015, out)

    out = np.clip(out, 0.0, cap)

    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out