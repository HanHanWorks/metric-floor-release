import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) else 105.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 温和去抖平滑
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = 0.72 * out + 0.28 * sm

    # 自适应收缩：低功率段保守收缩，高功率段轻微收缩
    tiny_mask = out < 0.02 * cap
    out[tiny_mask] = out[tiny_mask] * 0.35

    low_mask = (out >= 0.02 * cap) & (out < 0.08 * cap)
    out[low_mask] = out[low_mask] * 0.68

    mid_low_mask = (out >= 0.08 * cap) & (out < 0.20 * cap)
    out[mid_low_mask] = out[mid_low_mask] * 0.85

    mid_mask = (out >= 0.20 * cap) & (out < 0.45 * cap)
    out[mid_mask] = out[mid_mask] * 0.93

    mid_high_mask = (out >= 0.45 * cap) & (out <= 0.80 * cap)
    out[mid_high_mask] = out[mid_high_mask] * 0.975

    high_mask = out > 0.80 * cap
    out[high_mask] = 0.80 * cap + (out[high_mask] - 0.80 * cap) * 0.82

    # 抑制极端尖峰（相对局部中位数）
    if n >= 5:
        med = np.median(out)
        spike = out > (med + 0.28 * cap)
        out[spike] = 0.85 * out[spike] + 0.15 * med

    # 抑制异常下跌尖峰
    if n >= 5:
        med = np.median(out)
        dip = (out < med - 0.22 * cap) & (out < 0.10 * cap)
        out[dip] = 0.90 * out[dip] + 0.10 * med

    # 局部尖峰/凹陷抑制（基于滑动邻域）
    if n >= 7:
        padded = np.concatenate([out[:1], out, out[-1:]])
        local_med = np.median(np.stack([padded[i:i+n] for i in range(3)]), axis=0)
        hi = out > local_med + 0.22 * cap
        out[hi] = 0.88 * out[hi] + 0.12 * local_med[hi]
        lo = (out < local_med - 0.18 * cap) & (out < 0.12 * cap)
        out[lo] = 0.92 * out[lo] + 0.08 * local_med[lo]

    out = np.clip(out, 0.0, cap)

    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out