import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns and len(view) > 0 else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    # 基础：原始预测
    out = pp.copy()

    # 1) 轻度平滑：抑制高频抖动（15min 步自相关 0.998，说明信号很平滑，
    #    但 ramp_p99 仍有一定幅度，做很小的平滑即可，避免破坏峰值）
    if n >= 3:
        kernel = np.array([0.15, 0.70, 0.15])
        pad = np.pad(pp, (1, 1), mode='edge')
        smooth = np.convolve(pad, kernel, mode='valid')
        out = 0.85 * out + 0.15 * smooth

    # 2) 低出力区（大量样本 < 0.2cap）做轻微抬升/收缩到合理范围，
    #    避免负值和异常小值；同时保持非负
    out = np.clip(out, 0.0, cap)

    # 3) 对极低值做温和的“去噪抬升”，因为 frac_zero 很低(0.55%)，
    #    说明真实出力很少为 0，预测侧接近 0 的点更可能是噪声
    low_mask = out < 0.02 * cap
    if np.any(low_mask):
        out[low_mask] = 0.5 * out[low_mask] + 0.5 * (0.02 * cap)

    # 4) 日周期一致性：用当天前后点的局部中位数做轻微校正，
    #    仅对偏离局部中位数过大的点做收缩（防止尖刺）
    if n >= 5:
        win = 5
        pad = np.pad(out, (win // 2, win // 2), mode='edge')
        # 滑动中位数
        med = np.array([np.median(pad[i:i + win]) for i in range(n)])
        # 偏离过大时向中位数收缩
        dev = out - med
        thresh = 0.15 * cap
        big = np.abs(dev) > thresh
        if np.any(big):
            out[big] = med[big] + np.sign(dev[big]) * thresh

    # 5) 最终非负与容量约束
    out = np.clip(out, 0.0, cap)

    # 保证不是常数序列
    if np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out