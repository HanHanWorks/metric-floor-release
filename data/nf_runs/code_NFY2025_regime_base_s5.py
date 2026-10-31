import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 低出力段平滑（抑制抖动）
    low_mask = pp < 0.2 * cap
    if low_mask.any() and n >= 3:
        padded = np.pad(pp, (1, 1), mode='edge')
        smooth = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        out = np.where(low_mask, 0.5 * pp + 0.5 * smooth, out)

    # 中高出力段趋势小幅外推
    high_mask = pp > 0.6 * cap
    if high_mask.any() and n > 1:
        diff = np.diff(pp, prepend=pp[0])
        boost = np.clip(diff, -0.05 * cap, 0.05 * cap)
        out = np.where(high_mask, pp + 0.3 * boost, out)

    # 极低出力轻微抬升
    near_zero = (pp < 0.01 * cap) & (pp >= 0)
    out = np.where(near_zero, pp + 0.002 * cap, out)

    # 中段向日内趋势回归
    mid_mask = (pp >= 0.2 * cap) & (pp <= 0.6 * cap)
    if mid_mask.any() and n >= 5:
        padded = np.pad(pp, (2, 2), mode='edge')
        local_mean = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
        dev = pp - local_mean
        shrink = np.clip(dev, -0.08 * cap, 0.08 * cap)
        out = np.where(mid_mask, pp - 0.25 * shrink, out)

    # 相邻点跳变抑制（低-中段）
    if n >= 3:
        padded = np.pad(pp, (1, 1), mode='edge')
        neighbor_mean = (padded[:-2] + padded[2:]) / 2.0
        dev_nb = pp - neighbor_mean
        lowmid_mask = pp < 0.6 * cap
        jump = np.abs(dev_nb) > 0.03 * cap
        fix_mask = lowmid_mask & jump
        out = np.where(fix_mask, out - 0.2 * dev_nb, out)

    # 强连续性约束
    if n >= 5:
        padded = np.pad(out, (2, 2), mode='edge')
        ref = (0.15 * padded[:-4] + 0.35 * padded[1:-3] +
               0.35 * padded[3:-1] + 0.15 * padded[4:])
        dev_ref = out - ref
        band_mask = (pp < 0.6 * cap) & (np.abs(dev_ref) > 0.02 * cap) & (np.abs(dev_ref) < 0.10 * cap)
        out = np.where(band_mask, out - 0.15 * dev_ref, out)

    # 极低段去噪
    very_low = pp < 0.15 * cap
    if very_low.any() and n >= 3:
        padded = np.pad(out, (1, 1), mode='edge')
        sm = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
        out = np.where(very_low, 0.7 * out + 0.3 * sm, out)

    # 中低段去偏（中值）
    if n >= 7:
        core_mask = (pp >= 0.15 * cap) & (pp <= 0.5 * cap)
        if core_mask.any():
            padded = np.pad(out, (3, 3), mode='edge')
            windows = np.stack([padded[i:i + n] for i in range(7)], axis=0)
            med = np.median(windows, axis=0)
            dev_med = out - med
            med_mask = core_mask & (np.abs(dev_med) > 0.03 * cap) & (np.abs(dev_med) < 0.12 * cap)
            out = np.where(med_mask, out - 0.12 * dev_med, out)

    # 极高出力段保守约束
    very_high = pp > 0.85 * cap
    if very_high.any():
        out = np.where(very_high, np.minimum(out, pp + 0.02 * cap), out)

    # 爬坡段连续性增强
    if n >= 3:
        d = np.diff(pp, prepend=pp[0])
        ramp_mask = (np.abs(d) > 0.012 * cap) & (np.abs(d) < 0.05 * cap) & (pp < 0.7 * cap)
        if ramp_mask.any():
            padded = np.pad(out, (1, 1), mode='edge')
            lin = 0.5 * (padded[:-2] + padded[2:])
            out = np.where(ramp_mask, 0.85 * out + 0.15 * lin, out)

    # 9点长窗偏差收缩
    if n >= 9:
        padded = np.pad(out, (4, 4), mode='edge')
        w = np.zeros(n)
        for k in range(9):
            w += padded[k:k + n]
        ref9 = w / 9.0
        dev9 = out - ref9
        midband = (pp >= 0.1 * cap) & (pp <= 0.55 * cap) & (np.abs(dev9) > 0.05 * cap) & (np.abs(dev9) < 0.15 * cap)
        out = np.where(midband, out - 0.08 * dev9, out)

    # 低出力段主导样本轻微抬升
    if n >= 5:
        low_dom = (pp < 0.25 * cap) & (pp > 0.02 * cap)
        if low_dom.any():
            padded = np.pad(out, (2, 2), mode='edge')
            ref5 = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
            lift = np.clip(ref5 - out, 0.0, 0.015 * cap)
            out = np.where(low_dom, out + 0.25 * lift, out)

    # 极低出力避免贴地
    if n >= 3:
        tiny = (pp < 0.02 * cap) & (pp > 0.0)
        if tiny.any():
            padded = np.pad(pp, (1, 1), mode='edge')
            nb_max = np.maximum(padded[:-2], padded[2:])
            lift_mask = tiny & (nb_max > 0.03 * cap)
            out = np.where(lift_mask, out + 0.004 * cap, out)

    # 二阶差分抑制（低-中段）
    if n >= 5:
        padded = np.pad(out, (2, 2), mode='edge')
        lin_ref = 0.5 * (padded[:-4] + padded[4:])
        dev2 = out - lin_ref
        spike_mask = (pp < 0.6 * cap) & (np.abs(dev2) > 0.025 * cap) & (np.abs(dev2) < 0.09 * cap)
        out = np.where(spike_mask, out - 0.18 * dev2, out)

    # 低出力段局部极值削峰填谷
    if n >= 3:
        padded = np.pad(out, (1, 1), mode='edge')
        left = padded[:-2]
        right = padded[2:]
        mid_low = (pp >= 0.02 * cap) & (pp < 0.2 * cap)
        peak = mid_low & (out > left) & (out > right)
        trough = mid_low & (out < left) & (out < right)
        avg_nb = 0.5 * (left + right)
        out = np.where(peak, out - 0.12 * np.clip(out - avg_nb, 0.0, 0.03 * cap), out)
        out = np.where(trough, out + 0.12 * np.clip(avg_nb - out, 0.0, 0.03 * cap), out)

    # 低-中段主导样本，对残差做极轻的均值回归
    if n >= 5:
        dom_mask = (pp >= 0.05 * cap) & (pp <= 0.45 * cap)
        if dom_mask.any():
            padded = np.pad(out, (2, 2), mode='edge')
            loc = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
            resid = out - loc
            shrink_mask = dom_mask & (np.abs(resid) > 0.02 * cap) & (np.abs(resid) < 0.08 * cap)
            out = np.where(shrink_mask, out - 0.10 * resid, out)

    # 基于 step_autocorr 的温和一阶差分约束
    if n >= 3:
        d = np.diff(out, prepend=out[0])
        d_next = np.diff(out, append=out[-1])
        same_sign = (d * d_next > 0) & (np.abs(d) < 0.03 * cap) & (np.abs(d_next) < 0.03 * cap)
        if same_sign.any():
            mid_pt = np.zeros(n, dtype=bool)
            mid_pt[1:-1] = same_sign[1:-1]
            padded = np.pad(out, (1, 1), mode='edge')
            lin = 0.5 * (padded[:-2] + padded[2:])
            out = np.where(mid_pt, 0.92 * out + 0.08 * lin, out)

    # 低出力段主导样本的日内基线对齐
    if n >= 5:
        base_mask = (pp > 0.0) & (pp < 0.25 * cap)
        if base_mask.any():
            padded = np.pad(out, (2, 2), mode='edge')
            base5 = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
            dev_b = out - base5
            pull = base_mask & (np.abs(dev_b) > 0.015 * cap) & (np.abs(dev_b) < 0.06 * cap)
            out = np.where(pull, out - 0.12 * dev_b, out)

    # 中段小幅偏差收缩
    if n >= 5:
        mid2 = (pp >= 0.25 * cap) & (pp <= 0.55 * cap)
        if mid2.any():
            padded = np.pad(out, (2, 2), mode='edge')
            lm = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
            dev_m = out - lm
            m2 = mid2 & (np.abs(dev_m) > 0.02 * cap) & (np.abs(dev_m) < 0.07 * cap)
            out = np.where(m2, out - 0.10 * dev_m, out)

    # 低出力主导段（<0.2cap，占样本近七成）的极轻全局收缩
    if n >= 7:
        low_dom2 = (pp > 0.0) & (pp < 0.2 * cap)
        if low_dom2.any():
            padded = np.pad(out, (3, 3), mode='edge')
            windows = np.stack([padded[i:i + n] for i in range(7)], axis=0)
            med7 = np.median(windows, axis=0)
            dev_l = out - med7
            lmask = low_dom2 & (np.abs(dev_l) > 0.01 * cap) & (np.abs(dev_l) < 0.04 * cap)
            out = np.where(lmask, out - 0.08 * dev_l, out)

    # 低出力段（0.02~0.15cap）与相邻点均值的一致性微调
    if n >= 3:
        seg = (pp >= 0.02 * cap) & (pp < 0.15 * cap)
        if seg.any():
            padded = np.pad(out, (1, 1), mode='edge')
            nb = 0.5 * (padded[:-2] + padded[2:])
            dev_n = out - nb
            seg_mask = seg & (np.abs(dev_n) > 0.015 * cap) & (np.abs(dev_n) < 0.05 * cap)
            out = np.where(seg_mask, out - 0.10 * dev_n, out)

    # 极低出力段（<0.05cap，占样本近半）的稳定性增强
    if n >= 5:
        vlow = (pp >= 0.0) & (pp < 0.05 * cap)
        if vlow.any():
            padded = np.pad(out, (2, 2), mode='edge')
            w5 = np.stack([padded[i:i + n] for i in range(5)], axis=0)
            med5 = np.median(w5, axis=0)
            dev_v = out - med5
            vmask = vlow & (np.abs(dev_v) > 0.005 * cap) & (np.abs(dev_v) < 0.03 * cap)
            out = np.where(vmask, out - 0.10 * dev_v, out)

    # 中高段（0.4~0.7cap）轻微向局部均值收敛，抑制过度外推
    if n >= 5:
        mh = (pp >= 0.4 * cap) & (pp <= 0.7 * cap)
        if mh.any():
            padded = np.pad(out, (2, 2), mode='edge')
            lm5 = (padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]) / 5.0
            dev_mh = out - lm5
            mhmask = mh & (np.abs(dev_mh) > 0.02 * cap) & (np.abs(dev_mh) < 0.08 * cap)
            out = np.where(mhmask, out - 0.08 * dev_mh, out)

    # 新增：低出力段（<0.2cap，主导样本）向日内基线温和回归
    # 该段 day_autocorr=0.445 表明存在较强的日间基线结构，对残差做轻度向局部均值收缩可减少日内漂移。
    if n >= 9:
        low_base = (pp > 0.0) & (pp < 0.2 * cap)
        if low_base.any():
            padded = np.pad(out, (4, 4), mode='edge')
            w9 = np.zeros(n)
            for k in range(9):
                w9 += padded[k:k + n]
            ref9b = w9 / 9.0
            dev9b = out - ref9b
            bmask = low_base & (np.abs(dev9b) > 0.01 * cap) & (np.abs(dev9b) < 0.05 * cap)
            out = np.where(bmask, out - 0.06 * dev9b, out)

    # 新增：整体极轻向原始 PP 收缩，防止多步叠加后偏离预测基准过远
    # 通过限定在 |out-pp| 较小的区间，避免破坏已做的合理修正。
    if n >= 1:
        dev_pp = out - pp
        small_dev = (np.abs(dev_pp) > 0.005 * cap) & (np.abs(dev_pp) < 0.03 * cap)
        out = np.where(small_dev, out - 0.05 * dev_pp, out)

    # 物理约束
    out = np.clip(out, 0.0, cap)

    return out