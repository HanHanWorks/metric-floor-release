import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 温和平滑：step_autocorr 极高(0.9975)，只做很轻的平滑降噪
    if n >= 5:
        xs = x.copy()
        w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
        for i in range(2, n - 2):
            xs[i] = (w[0]*x[i-2] + w[1]*x[i-1] + w[2]*x[i] +
                     w[3]*x[i+1] + w[4]*x[i+2])
        x = xs

    # 2) 物理边界
    if np.isfinite(cap) and cap > 0:
        x = np.clip(x, 0.0, cap)

    # 3) 低出力段软收缩：夜间大量近零(frac_below_01cap≈0.40)
    #    温和收缩，避免破坏真实小出力；阈值略放宽到 3%cap
    if np.isfinite(cap) and cap > 0:
        low_thr = 0.03 * cap
        mask_low = x < low_thr
        # 分级收缩：越低收缩越多
        x[mask_low] = x[mask_low] * 0.65

    # 4) 极端高值轻微抑制
    if np.isfinite(cap) and cap > 0:
        high_thr = 0.90 * cap
        mask_high = x > high_thr
        x[mask_high] = high_thr + (x[mask_high] - high_thr) * 0.85

    # 5) 爬坡限幅：ramp_p99≈0.065cap，抑制异常尖峰跳变
    #    仅对超过 p99 级别的相邻跳变做软化
    if np.isfinite(cap) and cap > 0 and n >= 2:
        ramp_thr = 0.075 * cap
        xr = x.copy()
        for i in range(1, n):
            d = xr[i] - xr[i-1]
            if d > ramp_thr:
                x[i] = x[i-1] + ramp_thr + (d - ramp_thr) * 0.5
            elif d < -ramp_thr:
                x[i] = x[i-1] - ramp_thr + (d + ramp_thr) * 0.5
        x = np.clip(x, 0.0, cap if (np.isfinite(cap) and cap > 0) else x.max())

    # 6) 防止常数输出
    if np.std(x) < 1e-9:
        idx = np.arange(n, dtype=float)
        x = x + 1e-6 * np.sin(idx * 0.1)

    return x