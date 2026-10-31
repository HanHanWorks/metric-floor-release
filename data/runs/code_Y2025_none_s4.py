import numpy as np
import pandas as pd


def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 数据画像: step_autocorr=0.9968 极高（序列平滑）；pp_mean_ratio=0.2223 偏低
    # frac_below_02cap=0.5647 / frac_below_01cap=0.3317 → 低出力段占主导
    # 历史: 第50轮 ACC=55.65 为本轮基准，平滑+分段收缩结构较优
    # 本轮策略: 保持平滑主干，微调低段收缩与高段上修幅度，收紧变化率限制

    # 1) 5 点平滑（权重向中心集中，保信号）
    if n >= 5:
        sm5 = out.copy()
        sm5[2:-2] = (0.12 * out[:-4] + 0.20 * out[1:-3] + 0.36 * out[2:-2]
                     + 0.20 * out[3:-1] + 0.12 * out[4:])
        sm5[0] = out[0]
        sm5[1] = 0.5 * out[1] + 0.5 * out[2]
        sm5[-2] = 0.5 * out[-2] + 0.5 * out[-3]
        sm5[-1] = out[-1]
        out = 0.42 * out + 0.58 * sm5

    # 2) 7 点平滑（较弱权重，避免过度钝化）
    if n >= 7:
        sm7 = out.copy()
        sm7[3:-3] = (0.06 * out[:-6] + 0.11 * out[1:-5] + 0.15 * out[2:-4]
                     + 0.36 * out[3:-3]
                     + 0.15 * out[4:-2] + 0.11 * out[5:-1] + 0.06 * out[6:])
        sm7[0] = out[0]
        sm7[1] = 0.5 * out[1] + 0.5 * out[2]
        sm7[2] = 0.33 * out[2] + 0.34 * out[3] + 0.33 * out[4]
        sm7[-3] = 0.33 * out[-3] + 0.34 * out[-4] + 0.33 * out[-5]
        sm7[-2] = 0.5 * out[-2] + 0.5 * out[-3]
        sm7[-1] = out[-1]
        out = 0.64 * out + 0.36 * sm7

    # 3) 极低出力段收缩（< 0.10 cap，占 33%）
    very_low_mask = out < 0.10 * cap
    out = np.where(very_low_mask, out * 0.735, out)

    # 4) 低出力段收缩（0.10-0.20 cap）
    low_mask = (out >= 0.10 * cap) & (out < 0.20 * cap)
    out = np.where(low_mask, out * 0.815, out)

    # 5) 极小值抑制（< 0.01 cap）
    tiny_mask = out < 0.01 * cap
    out = np.where(tiny_mask, out * 0.22, out)

    # 6) 中低出力段轻微收缩（0.20-0.40 cap）
    mid_low_mask = (out >= 0.20 * cap) & (out < 0.40 * cap)
    out = np.where(mid_low_mask, out * 0.920, out)

    # 7) 中高出力段轻微收缩（0.40-0.60 cap）
    mid_high_mask = (out >= 0.40 * cap) & (out <= 0.60 * cap)
    out = np.where(mid_high_mask, out * 0.950, out)

    # 8) 高出力段温和上修（p99 约 0.70*cap，避免过度拔高）
    high_mask = out > 0.60 * cap
    out = np.where(high_mask, out * 1.070, out)

    # 9) 物理约束
    out = np.clip(out, 0.0, cap)

    # 10) 轻度变化率限制（ramp_p99≈0.0476*cap）
    if n >= 2:
        max_step = 0.050 * cap
        for i in range(1, n):
            diff = out[i] - out[i - 1]
            if diff > max_step:
                out[i] = out[i - 1] + max_step
            elif diff < -max_step:
                out[i] = out[i - 1] - max_step
        out = np.clip(out, 0.0, cap)

    # 11) 防止常数输出
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out