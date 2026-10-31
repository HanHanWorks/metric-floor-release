import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    x = pp / cap
    n = len(x)
    if n == 0:
        return x * cap

    # 1) 平滑（步自相关高，安全），但保留边缘
    if n >= 3:
        xs = x.copy()
        xs[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        xs[0] = 0.75 * x[0] + 0.25 * x[1]
        xs[-1] = 0.75 * x[-1] + 0.25 * x[-2]
    else:
        xs = x.copy()

    # 2) 低值区温和收缩：75% 样本 < 0.2cap，假阳性代价高
    #    用平滑过渡，避免中高值被过度压制
    low = np.clip(xs / 0.15, 0.0, 1.0)
    low_smooth = low ** 0.6
    shrink = 0.72 + 0.28 * low_smooth
    xs = xs * shrink

    # 3) 非负裁剪
    xs = np.clip(xs, 0.0, 1.0)

    # 4) 防止常数输出
    if np.std(xs) < 1e-6:
        idx = np.arange(n)
        xs = xs + 1e-4 * (0.5 + 0.5 * np.sin(idx * 0.1))

    return xs * cap