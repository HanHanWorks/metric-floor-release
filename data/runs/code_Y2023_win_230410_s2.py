import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    if not np.isfinite(cap) or cap <= 0:
        m = np.nanmax(pp) if np.isfinite(np.nanmax(pp)) else 0.0
        cap = m if m > 0 else 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()
    out = np.where(np.isfinite(out), out, 0.0)

    # 1) 去毛刺：中值滤波（仅修正尖峰，保护趋势与爬坡）
    if n >= 5:
        med = out.copy()
        for i in range(2, n - 2):
            med[i] = np.median(out[i-2:i+3])
        dev = np.abs(out - med)
        thr = max(0.02 * cap, np.nanpercentile(dev, 90) * 1.5)
        spike = dev > thr
        out[spike] = med[spike]

    # 2) 低功率段轻微抬升（缓解低值区系统性低估）
    ratio = out / cap
    low = ratio < 0.05
    out[low] = out[low] * 1.05 + 0.002 * cap

    # 3) 极低值处理
    zero_like = out < 0.001 * cap
    if np.all(zero_like):
        out = out + 0.01 * cap
    else:
        out[zero_like] = 0.0

    # 4) 边界约束
    out = np.clip(out, 0.0, cap)

    # 5) 轻量时序平滑，仅在平缓处（保护大爬坡）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        d = np.abs(np.diff(out, prepend=out[0]))
        thresh = np.nanpercentile(d, 80)
        smooth_mask = d < thresh
        out[smooth_mask] = sm[smooth_mask]

    # 6) 中高功率段轻微抑制（防高估）
    ratio = out / cap
    high = ratio > 0.85
    out[high] = out[high] * 0.985

    out = np.clip(out, 0.0, cap)

    # 7) 日周期轻微校准：通过时段均值做低幅校正，保持平滑
    if n >= 96:
        k = n // 96
        if k >= 1:
            prof = np.zeros(96)
            cnt = np.zeros(96)
            for i in range(k):
                seg = out[i*96:(i+1)*96]
                prof[:len(seg)] += seg
                cnt[:len(seg)] += 1
            valid = cnt > 0
            prof[valid] /= cnt[valid]
            # 时段均值与整体均值的偏差做温和校正
            gmean = np.mean(out)
            if gmean > 1e-6:
                adj = np.tile(prof, k)[:n]
                # 仅对平缓处应用，避免破坏爬坡
                d = np.abs(np.diff(out, prepend=out[0]))
                thr = np.nanpercentile(d, 70)
                calm = d < thr
                corr = 0.15 * (adj - gmean)
                out[calm] = out[calm] + corr[calm]
            out = np.clip(out, 0.0, cap)

    # 防止常数输出
    if np.nanstd(out) < 1e-9:
        out = out + 0.01 * cap * np.sin(np.arange(n) * 2 * np.pi / max(n, 1))

    return out