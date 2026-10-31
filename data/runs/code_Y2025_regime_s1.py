import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view else 186.725
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    
    out = pp.copy()
    n = len(out)
    
    ratio = np.where(cap > 0, pp / cap, 0.0)
    
    # 全局轻微收缩：低出力占59%，整体偏保守更安全
    out = out * 0.97
    
    # 低出力段 (<20%Cap)：主要误差来源，非线性收缩
    low_mask = pp < 0.20 * cap
    low_shrink = 0.90 + 0.10 * np.clip(ratio / 0.20, 0, 1)  # 0.90~1.00
    out = np.where(low_mask, out * low_shrink, out)
    
    # 低段夜间极低出力轻微抬升，避免系统性低估
    night_mask = low_mask & ((hour < 5) | (hour >= 21))
    out = np.where(night_mask & (ratio < 0.08), out * 1.04, out)
    
    # 中出力段 (20-60%Cap)：三点加权平滑抑制抖动
    mid_mask = (pp >= 0.20 * cap) & (pp < 0.60 * cap)
    if np.any(mid_mask):
        smoothed = out.copy()
        for i in range(1, n - 1):
            if mid_mask[i]:
                smoothed[i] = 0.2 * out[i-1] + 0.6 * out[i] + 0.2 * out[i+1]
        out = smoothed
    
    # 强爬坡段：中值滤波抗孤立尖峰
    if n >= 3:
        ramp = np.abs(np.diff(out, prepend=out[0]))
        ramp_mask = ramp > 0.12 * cap
        if np.any(ramp_mask):
            med = out.copy()
            for i in range(1, n - 1):
                if ramp_mask[i]:
                    med[i] = np.median([out[i-1], out[i], out[i+1]])
            out = med
    
    # 高出力段 (>60%Cap)：做极轻微抬升补偿整体收缩
    high_mask = pp > 0.60 * cap
    out = np.where(high_mask, out * 1.03, out)
    
    # 新增：低出力段内的日内一致性平滑（低段占59%，进一步压低抖动）
    # 对低段占主导的日子，做弱化的相邻点混合以稳定序列
    if n >= 3:
        low_frac = np.mean(low_mask)
        if low_frac > 0.5:
            cons = out.copy()
            for i in range(1, n - 1):
                if low_mask[i] and low_mask[i-1] and low_mask[i+1]:
                    cons[i] = 0.15 * out[i-1] + 0.7 * out[i] + 0.15 * out[i+1]
            out = cons
    
    out = np.clip(out, 0, cap)
    return out