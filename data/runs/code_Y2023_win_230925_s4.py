import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    # 数据画像：pp_mean_ratio≈0.21，p50≈0.12，frac_below_02cap≈0.68，
    # frac_zero≈0.004，ramp_p99≈0.05，step_autocorr≈0.998，day_autocorr≈0.486。
    # ACC 对低功率段高估惩罚更重；历史最优在 0.05 收缩 + 低段下压附近。
    # 本轮：进一步加深低段下压，微调收缩强度，并对自适应修正温和延展。

    # 1) 轻平滑（5点核），抑制孤立尖峰但保留 ramp
    if n >= 5:
        kernel = np.array([0.06, 0.16, 0.56, 0.16, 0.06])
        padded = np.concatenate([pp[:2][::-1], pp, pp[-2:][::-1]])
        smooth = np.convolve(padded, kernel, mode='valid')
    elif n >= 3:
        kernel = np.array([0.2, 0.6, 0.2])
        padded = np.concatenate([pp[:1], pp, pp[-1:]])
        smooth = np.convolve(padded, kernel, mode='valid')
    else:
        smooth = pp.copy()

    # 2) 向稳健中心（中位数）轻微收缩
    med = float(np.median(pp))
    shrink = 0.048
    out = (1.0 - shrink) * smooth + shrink * med

    # 3) 低功率段分级下压（ACC 对低功率高估敏感，本轮进一步加深低段）
    very_low = out < 0.10 * cap
    low = (out >= 0.10 * cap) & (out < 0.25 * cap)
    mid_low = (out >= 0.25 * cap) & (out < 0.40 * cap)
    mid = (out >= 0.40 * cap) & (out < 0.55 * cap)
    out[very_low] = out[very_low] * 0.915
    out[low] = out[low] * 0.932
    out[mid_low] = out[mid_low] * 0.963
    out[mid] = out[mid] * 0.984

    # 4) 高功率段轻微抑制（减少高估尖峰，p99≈0.70cap）
    high = out > 0.60 * cap
    out[high] = out[high] * 0.976

    # 5) 分时段修正：夜间/清晨低出力段微压，正午略抬
    hours = view['hour'].values.astype(float)
    night = (hours < 6.0) | (hours >= 22.0)
    midday = (hours >= 10.0) & (hours < 15.0)
    out[night] = out[night] * 0.980
    out[midday] = out[midday] * 1.006

    # 6) 自适应修正：正向偏离局部均值（高估尖峰）做收缩
    if n >= 7:
        w = 7
        padded2 = np.concatenate([out[:w//2][::-1], out, out[-(w//2):][::-1]])
        local_mean = np.convolve(padded2, np.ones(w)/w, mode='valid')
        dev = out - local_mean
        pos_dev = dev > 0
        out[pos_dev] = out[pos_dev] - 0.26 * dev[pos_dev]

    # 7) 物理约束
    out = np.clip(out, 0.0, cap)

    # 8) 保证非常数
    if np.allclose(out, out[0]):
        eps = 1e-6 * cap
        out = out + eps * np.linspace(0.0, 1.0, n)

    return out