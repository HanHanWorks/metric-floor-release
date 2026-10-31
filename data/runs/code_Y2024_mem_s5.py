import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    x = np.clip(pp, 0.0, None)

    # 1) 轻度平滑，保留爬坡结构（更温和，避免过度平滑损失峰值）
    if n >= 5:
        xp = np.pad(x, (2, 2), mode='edge')
        sm = 0.15 * xp[:-4] + 0.20 * xp[1:-3] + 0.30 * xp[2:-2] + 0.20 * xp[3:-1] + 0.15 * xp[4:]
    elif n >= 3:
        xp = np.pad(x, (1, 1), mode='edge')
        sm = 0.25 * xp[:-2] + 0.5 * xp[1:-1] + 0.25 * xp[2:]
    else:
        sm = x.copy()

    # 2) 低出力收缩（M06）：低段按比温和收缩，抑制噪声性抖动
    if np.isfinite(cap) and cap > 0:
        r = sm / cap
        shrink = np.ones_like(r)
        shrink = np.where(r < 0.05, 0.72, shrink)
        shrink = np.where((r >= 0.05) & (r < 0.15), 0.86, shrink)
        shrink = np.where((r >= 0.15) & (r < 0.25), 0.95, shrink)
        sm = sm * shrink

    # 3) 容量约束/削顶（M07）
    if np.isfinite(cap) and cap > 0:
        sm = np.clip(sm, 0.0, cap)

    # 4) 防止常数输出
    if np.allclose(sm, sm[0]):
        return x

    return sm