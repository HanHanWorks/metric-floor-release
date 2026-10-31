import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0

    if has_cap:
        ratio = out / cap
    else:
        denom = np.nanmax(out) + 1e-9
        ratio = out / denom

    # 数据画像：
    # - frac_below_01cap=0.52, frac_below_02cap=0.68 => 大部分时间低出力
    # - frac_zero 极低(0.0018) => 真实出力很少严格为0，不应过度压0
    # - ramp_p99=0.0385, step_autocorr=0.9985 => 序列平滑，可做平滑但别过头
    # - day_autocorr=0.7245 => 日间有相关性
    # 历史表现显示：轻度整体收缩 + 低段平滑最稳，过度分档收缩会拉低 ACC

    # 0) 极低出力段：仅轻微压缩，不强制为0
    tiny_thr = 0.005 * (cap if has_cap else denom)
    out = np.where(out < tiny_thr, out * 0.75, out)

    # 1) 平滑（低出力段更平滑，抑制噪声；对ACC相对误差敏感区温和处理）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        low_mask = ratio < 0.20
        out = np.where(low_mask, 0.6 * sm + 0.4 * out, out)

    # 2) 全局温和收缩：日前预测整体略偏高（pp_mean_ratio 低但低段偏高）
    # 统一轻微收缩以提升整体 ACC，避免分档切换带来的跳变
    out = out * 0.95

    # 3) 中低出力段（20%-45% cap）：日前预测此段常偏高，温和再收缩
    mid_low = (ratio >= 0.20) & (ratio < 0.45)
    out = np.where(mid_low, out * 0.97, out)

    # 4) 中高出力段（45%-85% cap）：轻平滑抑制噪声
    if n >= 3:
        mid_high = (ratio >= 0.45) & (ratio <= 0.85)
        if np.any(mid_high):
            ms = out.copy()
            ms[1:-1] = 0.12 * out[:-2] + 0.76 * out[1:-1] + 0.12 * out[2:]
            out = np.where(mid_high, ms, out)

    # 5) 高出力段（>85% cap）：保留峰值，轻平滑
    if n >= 3:
        high_mask = ratio > 0.85
        if np.any(high_mask):
            hs = out.copy()
            hs[1:-1] = 0.15 * out[:-2] + 0.7 * out[1:-1] + 0.15 * out[2:]
            out = np.where(high_mask, hs, out)

    # 6) 物理约束
    if has_cap:
        out = np.clip(out, 0.0, cap)
    else:
        out = np.clip(out, 0.0, None)

    # 7) 兜底：常数输出时回退
    if np.allclose(out, out[0]):
        out = pp.copy()

    return out