import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)

    n = len(pp)
    out = pp.copy()
    ratio = pp / max(cap, 1e-6)

    low_mask = ratio < 0.20
    mid_mask = (ratio >= 0.20) & (ratio <= 0.60)
    high_mask = ratio > 0.60

    if n > 1:
        diff = np.abs(np.diff(pp, prepend=pp[0]))
    else:
        diff = np.zeros(n)
    ramp_mask = diff > 0.10 * cap

    # ---------- 低出力段 (<20%Cap, 占59%) ----------
    # 收缩压低相对误差，向零轻压；不做日历修正（M11 被拒）。
    if np.any(low_mask):
        r = np.clip(ratio / 0.20, 0.0, 1.0)
        # 极低出力收缩更强，接近边界几乎不收缩
        shrink = 0.70 + 0.30 * (r ** 1.6)
        low_out = pp * shrink
        out = np.where(low_mask, low_out, out)

    # ---------- 中出力段 (20-60%Cap, 占32%) ----------
    # step_autocorr 极高，平滑为主 + 轻滞后混合（M02+M09 温和形式）。
    if np.any(mid_mask) and n >= 3:
        padded = np.concatenate([[pp[0]], pp, [pp[-1]]])
        smooth = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        lag = np.concatenate([[pp[0]], pp[:-1]])
        mixed = 0.88 * smooth + 0.12 * lag
        out = np.where(mid_mask, mixed, out)

    # ---------- 强爬坡段 ----------
    # 中值滤波抗孤立尖峰，与当前结果轻混合（低段不适用）。
    if np.any(ramp_mask) and n >= 3:
        padded = np.concatenate([[pp[0]], pp, [pp[-1]]])
        med = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0), axis=0)
        ramp_apply = ramp_mask & (~low_mask)
        ramp_out = 0.40 * med + 0.60 * out
        out = np.where(ramp_apply, ramp_out, out)

    # ---------- 高出力段 (>60%Cap) ----------
    # 极轻微分位拉伸，幅度保守避免过冲。
    if np.any(high_mask):
        stretched = 0.6 * cap + (pp - 0.6 * cap) * 1.005
        out = np.where(high_mask, stretched, out)

    out = np.clip(out, 0.0, cap)

    if n > 1 and np.allclose(out, out[0]):
        out = pp + 1e-6 * np.arange(n)

    return out