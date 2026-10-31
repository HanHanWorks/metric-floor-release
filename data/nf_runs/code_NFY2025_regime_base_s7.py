import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
    dow = view['dow'].values.astype(float) if 'dow' in view.columns else None
    n = len(pp)
    out = pp.copy()

    # 1) 低出力段：轻度平滑抑制毛刺，保留零值结构
    low_mask = pp < 0.20 * cap
    if low_mask.sum() > 5:
        sm = pp.copy()
        for i in range(1, n - 1):
            if low_mask[i]:
                sm[i] = 0.5 * pp[i] + 0.25 * pp[i - 1] + 0.25 * pp[i + 1]
        out = np.where(low_mask, sm, out)

    # 2) 中高出力段：限幅异常大跳变（放宽限幅以减少对真实爬坡的抑制）
    mid_high = pp >= 0.20 * cap
    if mid_high.sum() > 5:
        lim = 0.08 * cap
        clipped = pp.copy()
        for i in range(1, n):
            d = clipped[i] - clipped[i - 1]
            if abs(d) > lim:
                clipped[i] = clipped[i - 1] + np.sign(d) * lim
        out = np.where(mid_high, clipped, out)

    # 3) 日周期形状校正（仅高段）
    if hour is not None:
        high_mask = pp > 0.60 * cap
        if high_mask.sum() > 10:
            df = pd.DataFrame({'h': np.floor(hour).astype(int), 'r': pp / cap})
            med = df.groupby('h')['r'].median()
            global_med = float(np.median(pp / cap))
            adj = np.array([med.get(int(h), global_med) for h in np.floor(hour).astype(int)])
            shaped = (0.85 * (pp / cap) + 0.15 * adj) * cap
            out = np.where(high_mask, shaped, out)

    # 4) 中低出力段(0.05~0.35cap)按 hour 形状做轻微收缩校正
    if hour is not None:
        midlow_mask = (pp >= 0.05 * cap) & (pp < 0.35 * cap)
        if midlow_mask.sum() > 30:
            h_int = np.floor(hour).astype(int)
            df2 = pd.DataFrame({'h': h_int, 'r': pp / cap})
            med2 = df2.groupby('h')['r'].median()
            g2 = float(np.median(pp[midlow_mask] / cap)) if midlow_mask.sum() > 0 else 0.2
            adj2 = np.array([med2.get(int(h), g2) for h in h_int])
            shaped2 = (0.90 * (pp / cap) + 0.10 * adj2) * cap
            out = np.where(midlow_mask, shaped2, out)

    # 5) 按 (hour, dow) 的稳健中位数做极轻微收缩，仅在高段
    if hour is not None and dow is not None:
        high_mask2 = pp > 0.55 * cap
        if high_mask2.sum() > 40:
            h_int = np.floor(hour).astype(int)
            d_int = dow.astype(int)
            df3 = pd.DataFrame({'h': h_int, 'd': d_int, 'r': pp / cap})
            grp = df3.groupby(['h', 'd'])['r'].median()
            g3 = float(np.median(pp / cap))
            keys = list(zip(h_int, d_int))
            adj3 = np.array([grp.get(k, g3) for k in keys])
            shaped3 = (0.93 * (pp / cap) + 0.07 * adj3) * cap
            out = np.where(high_mask2, shaped3, out)

    # 7) 极低出力段(<0.05cap) 轻度向上抬升，缓解系统性低估
    very_low = pp < 0.05 * cap
    if very_low.sum() > 20:
        nonzero = very_low & (pp > 0)
        bump = np.where(nonzero, pp + 0.03 * (0.05 * cap - pp), pp)
        out = np.where(very_low, np.minimum(bump, 0.05 * cap), out)

    # 8) 中间出力段(0.35~0.55cap) 轻微平滑，抑制序列抖动
    mid = (pp >= 0.35 * cap) & (pp <= 0.55 * cap)
    if mid.sum() > 20:
        sm2 = pp.copy()
        for i in range(1, n - 1):
            if mid[i]:
                sm2[i] = 0.7 * pp[i] + 0.15 * pp[i - 1] + 0.15 * pp[i + 1]
        out = np.where(mid, sm2, out)

    # 9) 过渡段 0.20~0.35cap 按 hour 形状做中等强度校正
    if hour is not None:
        trans_mask = (pp >= 0.20 * cap) & (pp < 0.35 * cap)
        if trans_mask.sum() > 30:
            h_int = np.floor(hour).astype(int)
            df4 = pd.DataFrame({'h': h_int, 'r': pp / cap})
            med4 = df4.groupby('h')['r'].median()
            g4 = float(np.median(pp[trans_mask] / cap))
            adj4 = np.array([med4.get(int(h), g4) for h in h_int])
            shaped4 = (0.88 * (pp / cap) + 0.12 * adj4) * cap
            out = np.where(trans_mask, shaped4, out)

    # 10) 高段(0.55~0.75cap) 轻度上抬
    upper_mid = (pp >= 0.55 * cap) & (pp < 0.75 * cap)
    if upper_mid.sum() > 20:
        out = np.where(upper_mid, pp + 0.015 * (0.75 * cap - pp), out)

    # 11) 新增：0.75~0.90cap 顶部段按 (hour) 形状轻度校正
    #     顶部段此前只在 clip 与 M05/ps 收缩间接影响，缺乏针对性形状对齐；
    #     该段对 ACC 贡献权重高，用 10% 中位数形状做温和贴合，避免过冲
    if hour is not None:
        top_mask = (pp >= 0.75 * cap) & (pp <= 0.90 * cap)
        if top_mask.sum() > 20:
            h_int = np.floor(hour).astype(int)
            df5 = pd.DataFrame({'h': h_int, 'r': pp / cap})
            med5 = df5.groupby('h')['r'].median()
            g5 = float(np.median(pp[top_mask] / cap))
            adj5 = np.array([med5.get(int(h), g5) for h in h_int])
            shaped5 = (0.90 * (pp / cap) + 0.10 * adj5) * cap
            out = np.where(top_mask, shaped5, out)

    # 12) 新增：极顶部(>0.90cap) 做极轻微下压收缩，抑制峰值过冲
    #     日前预测在接近满发时经常高估，导致溢出误差；温和收缩 2%
    extreme = pp > 0.90 * cap
    if extreme.sum() > 5:
        out = np.where(extreme, pp - 0.02 * (pp - 0.90 * cap), out)

    # 边界约束
    out = np.clip(out, 0.0, cap)
    if np.allclose(out, out[0]):
        out = pp.copy()
    return out