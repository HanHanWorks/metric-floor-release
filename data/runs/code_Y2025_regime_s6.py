import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    if not np.isfinite(cap) or cap <= 0:
        cap = np.nanmax(pp) if np.nanmax(pp) > 0 else 1.0

    out = pp.copy()
    n = len(pp)
    if n == 0:
        return out

    ratio = pp / cap
    low = ratio < 0.20
    mid = (ratio >= 0.20) & (ratio <= 0.60)
    high = ratio > 0.60

    ramp = np.zeros(n)
    if n > 1:
        ramp[1:] = np.abs(np.diff(pp)) / cap
    strong_ramp = ramp > 0.10

    # ---------- 低出力段 (<20%Cap): 占59% ----------
    # 温和收缩压向零，压低相对误差；低段轻度平滑抑制抖动
    low_out = pp * 0.86
    if n >= 3:
        sm = np.empty(n)
        sm[0] = pp[0]
        sm[-1] = pp[-1]
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        low_out = 0.50 * low_out + 0.50 * sm
    out = np.where(low, low_out, out)

    # ---------- 中出力段 (20-60%Cap): 占32% ----------
    if n >= 3:
        sm = np.empty(n)
        sm[0] = pp[0]
        sm[-1] = pp[-1]
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        out = np.where(mid, 0.70 * out + 0.30 * sm, out)

    # 中段与昨日同时刻轻度混合
    if n > 96:
        lag = np.empty(n)
        lag[:96] = pp[:96]
        lag[96:] = pp[:-96]
        out = np.where(mid, 0.88 * out + 0.12 * lag, out)

    # ---------- 强爬坡段 (|Δ|>10%Cap) ----------
    if n >= 3:
        sm = np.empty(n)
        sm[0] = pp[0]
        sm[-1] = pp[-1]
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        out = np.where(strong_ramp, 0.78 * out + 0.22 * sm, out)

    if n >= 3:
        med = np.empty(n)
        med[0] = pp[0]
        med[-1] = pp[-1]
        for i in range(1, n - 1):
            med[i] = np.median(pp[i-1:i+2])
        out = np.where(strong_ramp, 0.88 * out + 0.12 * med, out)

    # ---------- 高出力段 (>60%Cap): 轻度保留原值 ----------
    # 高出力段基本保持原预测（避免损害），仅做极轻平滑
    if n >= 3:
        sm = np.empty(n)
        sm[0] = pp[0]
        sm[-1] = pp[-1]
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        out = np.where(high, 0.92 * out + 0.08 * sm, out)

    out = np.clip(out, 0.0, cap)
    return out