import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑：3 点加权，抑制 15 分钟噪声但保留爬坡
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 2) 低功率段收缩（针对高比例低功率场景，进一步收缩低段以降低绝对误差）
    if not np.isnan(cap) and cap > 0:
        ratio = out / cap
        shrink = np.ones_like(out)
        shrink = np.where(ratio < 0.03, 0.58, shrink)
        shrink = np.where((ratio >= 0.03) & (ratio < 0.06), 0.66, shrink)
        shrink = np.where((ratio >= 0.06) & (ratio < 0.10), 0.74, shrink)
        shrink = np.where((ratio >= 0.10) & (ratio < 0.20), 0.83, shrink)
        shrink = np.where((ratio >= 0.20) & (ratio < 0.30), 0.895, shrink)
        shrink = np.where((ratio >= 0.30) & (ratio < 0.45), 0.942, shrink)
        shrink = np.where((ratio >= 0.45) & (ratio < 0.60), 0.978, shrink)
        out = out * shrink

    # 2b) 极低段强制归零（frac_below_01cap 高，微小值对绝对误差贡献小但会造成假阳性）
    if not np.isnan(cap) and cap > 0:
        tiny = out < (0.006 * cap)
        out = np.where(tiny, 0.0, out)

    # 2c) 对低功率段再做一次温和收缩，进一步降低绝对误差
    if not np.isnan(cap) and cap > 0:
        ratio2 = out / cap
        shrink2 = np.ones_like(out)
        shrink2 = np.where(ratio2 < 0.05, 0.83, shrink2)
        shrink2 = np.where((ratio2 >= 0.05) & (ratio2 < 0.12), 0.885, shrink2)
        shrink2 = np.where((ratio2 >= 0.12) & (ratio2 < 0.25), 0.945, shrink2)
        out = out * shrink2

    # 2d) 高功率段轻微收缩（p99=0.754，高段样本稀少，防止过冲）
    if not np.isnan(cap) and cap > 0:
        ratio3 = out / cap
        shrink3 = np.ones_like(out)
        shrink3 = np.where((ratio3 >= 0.60) & (ratio3 < 0.80), 0.978, shrink3)
        shrink3 = np.where(ratio3 >= 0.80, 0.965, shrink3)
        out = out * shrink3

    # 2e) 爬坡限幅：依据 ramp_p99≈0.031*cap 的统计约束相邻点变化
    if not np.isnan(cap) and cap > 0 and n >= 2:
        max_step = 0.045 * cap
        for i in range(1, n):
            d = out[i] - out[i - 1]
            if d > max_step:
                out[i] = out[i - 1] + max_step
            elif d < -max_step:
                out[i] = out[i - 1] - max_step

    # 3) 非负裁剪
    out = np.clip(out, 0.0, None if np.isnan(cap) else cap)

    # 4) 防止常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out