import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    if not np.isfinite(cap) or cap <= 0:
        cap = np.nanmax(pp) if len(pp) else 1.0
        if not np.isfinite(cap) or cap <= 0:
            cap = 1.0

    out = pp.copy()
    n = len(out)
    if n == 0:
        return out

    low_thr = 0.20 * cap
    mid_thr = 0.60 * cap

    # 3点中值滤波（抗孤立尖峰）
    med = out.copy()
    if n >= 3:
        med[1:-1] = np.median(np.stack([out[:-2], out[1:-1], out[2:]], axis=0), axis=0)

    # 3点加权平滑
    sm = out.copy()
    if n >= 3:
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]

    # 低出力段 (<20%Cap, 占59%)：低出力收缩（M06）——按比例压向零，压低相对误差
    low_mask = pp < low_thr
    shrink = np.clip(pp / np.maximum(low_thr, 1e-6), 0.0, 1.0)
    shrink_coef = 0.85 + 0.10 * shrink
    low_out = shrink_coef * (0.75 * med + 0.25 * sm)
    out = np.where(low_mask, low_out, out)

    # 中出力段 (20-60%Cap, 占32%)：平滑为主，混合中值
    mid_mask = (pp >= low_thr) & (pp < mid_thr)
    out = np.where(mid_mask, 0.65 * sm + 0.35 * med, out)

    # 强爬坡段：中值滤波抗尖峰
    ramp = np.zeros(n)
    if n >= 2:
        ramp[1:] = np.abs(np.diff(pp))
    ramp_thr = 0.10 * cap
    ramp_mask = ramp > ramp_thr
    out = np.where(ramp_mask, 0.5 * med + 0.5 * out, out)

    # 高出力段 (>60%Cap)：分段映射（M12）——轻微压缩避免过冲
    high_mask = pp >= mid_thr
    if np.any(high_mask):
        high_out = 0.92 * pp + 0.08 * mid_thr
        out = np.where(high_mask, high_out, out)

    # 低出力段再叠加日历修正（M11）——仅限低段
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        h_rad = 2.0 * np.pi * hour / 24.0
        if np.any(low_mask):
            y = pp[low_mask]
            s = np.sin(h_rad[low_mask])
            c = np.cos(h_rad[low_mask])
            A = np.stack([np.ones_like(s), s, c], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                fit = coef[0] + coef[1] * np.sin(h_rad) + coef[2] * np.cos(h_rad)
                cal_out = 0.85 * out + 0.15 * fit
                out = np.where(low_mask, cal_out, out)
            except Exception:
                pass

    # 新增：低出力段温和抬底 + 保底（M05 小幅版，仅在极低段，避免破坏中段）
    # 用非常温和的阈值（3%Cap）消除接近零的抖动，且仅对低段生效
    floor_thr = 0.03 * cap
    tiny_mask = (pp < floor_thr) & low_mask
    if np.any(tiny_mask):
        out = np.where(tiny_mask, np.maximum(out, 0.4 * pp + 0.6 * floor_thr * 0.5), out)

    # 新增：低段整体轻微缩放（M01 局部版），仅在低段且非爬坡段施加
    # 目标：低段整体误差以相对误差为主，轻微下调可降低相对误差
    lowcal = low_mask & (~ramp_mask)
    if np.any(lowcal):
        out = np.where(lowcal, 0.97 * out, out)

    # 新增：中段与爬坡段结合——在爬坡末尾（前值低、当前转高）做轻微抬升
    # 识别由低向中跃迁的点，给予平滑过渡倾向
    if n >= 2:
        prev = np.concatenate([[pp[0]], pp[:-1]])
        rise = (pp - prev) > 0.05 * cap
        rise_mask = rise & (pp >= low_thr) & (pp < mid_thr)
        if np.any(rise_mask):
            out = np.where(rise_mask, 0.85 * out + 0.15 * sm, out)

    out = np.clip(out, 0.0, None)
    return out