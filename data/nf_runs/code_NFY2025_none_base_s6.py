import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 极轻平滑：仅对孤立尖刺做中值替换（step_autocorr=0.997 说明信号本身已很平滑，
    #    过度平滑会损失真实爬坡信息，反而降低 ACC）
    if n >= 5:
        med = out.copy()
        med[1:-1] = np.median(np.stack([out[:-2], out[1:-1], out[2:]]), axis=0)
        # 仅当与中值偏差较大时才替换（去尖刺，不动正常点）
        dev = np.abs(out - med)
        spike = dev > 0.03 * cap
        out = np.where(spike, 0.5 * out + 0.5 * med, out)

    # 2) 低值段：frac_below_01cap=0.297，夜间大量近零值。
    #    对极低值做轻微向 0 收缩（考核按容量归一，小误差在低值段被放大）
    low_mask = out < 0.03 * cap
    out[low_mask] = out[low_mask] * 0.97

    # 3) 高值段：p99=0.74cap，frac_above_09cap 仅 0.2%。
    #    对接近容量的点温和抑制，防止过预测
    high_mask = out > 0.80 * cap
    out[high_mask] = 0.80 * cap + (out[high_mask] - 0.80 * cap) * 0.90

    # 4) 爬坡限幅：ramp_max=0.297 异常，但 p99=0.051cap。
    #    用较宽松阈值（0.12cap）仅削极端跳变，保留正常爬坡
    max_ramp = 0.12 * cap
    for i in range(1, n):
        d = out[i] - out[i - 1]
        if d > max_ramp:
            out[i] = out[i - 1] + max_ramp
        elif d < -max_ramp:
            out[i] = out[i - 1] - max_ramp

    # 5) 非负裁剪
    out = np.clip(out, 0.0, cap)

    if np.std(out) < 1e-6:
        return pp

    return out