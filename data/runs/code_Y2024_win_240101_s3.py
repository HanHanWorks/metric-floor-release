import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    n = len(pp)
    if n == 0:
        return pp
    if 'Cap' in view.columns and len(view) > 0:
        cap = float(view['Cap'].iloc[0])
    else:
        cap = 85.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 85.0

    out = pp.copy()

    # 1) 温和去尖刺：仅在明显离群时向中位数收缩（保持原逻辑，稍紧阈值）
    if n >= 5:
        s = pd.Series(pp)
        med = s.rolling(5, center=True, min_periods=1).median().values
        diff = pp - med
        spike = np.abs(diff) > 0.11 * cap
        out = np.where(spike, 0.75 * pp + 0.25 * med, out)

    # 2) 极轻平滑
    if n >= 3:
        s2 = pd.Series(out)
        smooth = s2.rolling(3, center=True, min_periods=1).mean().values
        out = 0.95 * out + 0.05 * smooth

    # 3) 低出力段轻度收缩
    low = out < 0.05 * cap
    out = np.where(low, out * 0.96, out)

    # 4) 极低出力段进一步收缩
    verylow = out < 0.01 * cap
    out = np.where(verylow, out * 0.88, out)

    # 5) 日间结构一致性修正（仅对显著偏离才调整）
    if n >= 96 * 3:
        s3 = pd.Series(out)
        ref_prev = s3.shift(96).values
        ref_next = s3.shift(-96).values
        ref = np.nanmean(np.vstack([ref_prev, ref_next]), axis=0)
        valid = ~np.isnan(ref)
        blend = np.zeros(n, dtype=bool)
        blend[valid] = np.abs(out[valid] - ref[valid]) > 0.18 * cap
        out = np.where(blend, 0.92 * out + 0.08 * ref, out)

    # 6) 高功率峰值轻微保留：避免过度收缩峰段（峰段通常误差大但不应抹平）
    high = out > 0.85 * cap
    out = np.where(high, np.minimum(out, cap), out)

    # 7) 边界裁剪
    out = np.clip(out, 0.0, cap)

    # 8) 防止常数输出
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out