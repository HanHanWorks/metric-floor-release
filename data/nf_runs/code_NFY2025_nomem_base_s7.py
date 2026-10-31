import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 序列极平滑(step_autocorr=0.998)，不做时间平滑，避免引入偏差
    # frac_below_02cap=0.62，低段过预测常见，主导ACC误差

    # 低功率段：温和向0收缩
    low_thr = 0.15 * cap
    low_mask = out < low_thr
    if np.any(low_mask):
        out[low_mask] = out[low_mask] * 0.92

    # 极低段（<0.05cap）进一步收缩，抑制噪声型小值
    very_low = out < 0.05 * cap
    if np.any(very_low):
        out[very_low] = out[very_low] * 0.80

    # 高段抑制：p99约0.8cap，超过0.85cap预测可能过冲
    high_thr = 0.85 * cap
    high_mask = out > high_thr
    if np.any(high_mask):
        out[high_mask] = high_thr + (out[high_mask] - high_thr) * 0.84

    # 中高段轻微收缩：0.6-0.85cap 之间轻微高估倾向
    mid_high_mask = (out >= 0.6 * cap) & (out <= high_thr)
    if np.any(mid_high_mask):
        out[mid_high_mask] = out[mid_high_mask] * 0.978

    # 非负裁剪
    out = np.clip(out, 0.0, cap)

    # 防止常数序列
    if np.allclose(out, out[0]):
        eps = 1e-6 * max(cap, 1.0)
        out = out + eps * np.sin(np.arange(n) * 0.1)

    return out