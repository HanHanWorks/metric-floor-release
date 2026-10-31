import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view else 1.0
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)

    out = pp.copy()
    n = len(pp)

    low = pp < 0.20 * cap
    mid = (pp >= 0.20 * cap) & (pp < 0.60 * cap)
    high = pp >= 0.60 * cap

    ramp = np.zeros(n, dtype=bool)
    if n > 1:
        d = np.abs(np.diff(pp, prepend=pp[0]))
        ramp = d > 0.10 * cap

    # ============ 低出力段（59%）============
    # 轻微收缩压向零，压低相对误差（不动强爬坡）
    out = np.where(low & ~ramp, out * 0.945, out)

    # 极低值轻抬底（范围更窄，更温和）
    very_low = pp < 0.02 * cap
    floor_val = 0.0025 * cap
    out = np.where(very_low & ~ramp, np.maximum(out, floor_val), out)

    # 低段三点平滑（保留边界）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = np.where(low & ~ramp, 0.55 * out + 0.45 * sm, out)

    # 低段按小时分组向组内中位数温和回归
    if low.sum() > 30:
        hb = np.clip(hour.astype(int), 0, 23)
        for b in range(24):
            m = low & ~ramp & (hb == b)
            if m.sum() > 6:
                med_b = np.median(out[m])
                out = np.where(m, out + 0.14 * (med_b - out), out)

    # 低段按 (is_weekend, 4h桶) 再回归一层
    if low.sum() > 60:
        bucket4 = np.clip((hour // 4).astype(int), 0, 5)
        is_we = (dow >= 5).astype(int)
        for wk in range(2):
            for b in range(6):
                m = low & ~ramp & (is_we == wk) & (bucket4 == b)
                if m.sum() > 8:
                    med_wb = np.median(out[m])
                    out = np.where(m, out + 0.09 * (med_wb - out), out)

    # ============ 中出力段（32%）============
    if n >= 3 and mid.sum() > 20:
        sm2 = out.copy()
        sm2[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = np.where(mid & ~ramp, 0.55 * out + 0.45 * sm2, out)

    if mid.sum() > 40:
        bucket4m = np.clip((hour // 4).astype(int), 0, 5)
        for b in range(6):
            m = mid & ~ramp & (bucket4m == b)
            if m.sum() > 8:
                med_mb = np.median(out[m])
                out = np.where(m, out + 0.10 * (med_mb - out), out)

    # 中段：按 hour 分组的轻度回归（补充 4h 桶粒度不足）
    if mid.sum() > 40:
        hbm = np.clip(hour.astype(int), 0, 23)
        for b in range(24):
            m = mid & ~ramp & (hbm == b)
            if m.sum() > 10:
                med_mh = np.median(out[m])
                out = np.where(m, out + 0.06 * (med_mh - out), out)

    # ============ 强爬坡段（单独处理，避免与低/中段重叠冲突）============
    if n >= 3 and ramp.sum() > 5:
        sm3 = out.copy()
        sm3[1:-1] = 0.3 * out[:-2] + 0.4 * out[1:-1] + 0.3 * out[2:]
        out = np.where(ramp, 0.55 * out + 0.45 * sm3, out)

    # ============ 高出力段（<10%）============
    # 轻微分位拉伸展开动态范围（只在非常高出力时启用）
    very_high = pp >= 0.75 * cap
    if very_high.sum() > 3:
        c = 0.82 * cap
        out = np.where(very_high & ~ramp, c + (out - c) * 1.03, out)

    # ============ 新增：轻度日间一致性修正（非爬坡、非极低）============
    # 用前向/后向相邻点做极轻量的局部一致性约束，减少孤立抖动
    if n >= 5:
        # 只在非爬坡、低/中段且非极低值上执行
        mask_cons = (~ramp) & (pp >= 0.01 * cap)
        if mask_cons.sum() > 10:
            # 中心点与邻域均值偏差过大时温和回拉（避免破坏有效峰值）
            center = out.copy()
            left = np.concatenate([out[:1], out[:-1]])
            right = np.concatenate([out[1:], out[-1:]])
            neigh_mean = 0.5 * (left + right)
            dev = center - neigh_mean
            # 仅处理相对局部均值的离群点，温和收缩偏差
            pull = 0.08
            out = np.where(mask_cons, center - pull * np.sign(dev) * np.minimum(np.abs(dev), 0.02 * cap), out)

    out = np.clip(out, 0.0, cap)
    return out