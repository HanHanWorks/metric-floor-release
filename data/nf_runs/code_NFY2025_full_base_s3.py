import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # M03 中值滤波：抑制孤立尖峰
    def median_filter(a, w=3):
        if w <= 1 or len(a) < w:
            return a
        pad = w // 2
        ap = np.pad(a, (pad, pad), mode='edge')
        out = np.empty(len(a))
        for i in range(len(a)):
            out[i] = np.median(ap[i:i + w])
        return out

    x = median_filter(x, 3)

    # M02 平滑
    def smooth(a, w=3):
        if w <= 1 or len(a) < w:
            return a
        k = np.ones(w) / w
        pad = w // 2
        ap = np.pad(a, (pad, pad), mode='edge')
        return np.convolve(ap, k, mode='valid')[:len(a)]

    x = smooth(x, 3)

    # M09 滞后混合
    if n > 1:
        prev = np.concatenate([[x[0]], x[:-1]])
        x = 0.84 * x + 0.16 * prev

    # M05 抬底
    if not np.isnan(cap) and cap > 0:
        r = x / cap
        lift = 0.005
        r = np.where(r < 0.10, r + lift * (1.0 - r / 0.10), r)
        x = r * cap

    # M12 分段映射
    if not np.isnan(cap) and cap > 0:
        r = x / cap
        r2 = np.where(r < 0.05, r * 1.10 + 0.002, r)
        r2 = np.where(r > 0.85, 0.85 + (r2 - 0.85) * 0.88, r2)
        x = r2 * cap

    # M10 分位拉伸
    if not np.isnan(cap) and cap > 0:
        r = x / cap
        r = np.clip(r, 0.0, 1.0)
        r = np.power(r, 0.95)
        x = r * cap

    # M08 爬坡限制
    if n > 1 and not np.isnan(cap) and cap > 0:
        max_ramp = 0.055 * cap
        for i in range(1, n):
            d = x[i] - x[i - 1]
            if d > max_ramp:
                x[i] = x[i - 1] + max_ramp
            elif d < -max_ramp:
                x[i] = x[i - 1] - max_ramp

    # 日内形状对齐
    if not np.isnan(cap) and cap > 0 and n >= 96:
        days = n // 96
        xr = x / cap
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = xr[s:e]
            seg_max = np.max(seg)
            seg_mean = np.mean(seg)
            if seg_max > 1e-6:
                denom = max(seg_max, 1e-6)
                shape = seg / denom
                seg2 = seg * (0.98 + 0.02 * shape) + 0.002 * (seg_mean - seg) * (1.0 - shape)
                xr[s:e] = seg2
        x = xr * cap

    # M09 二次滞后混合（更弱权重）
    if n > 1:
        prev2 = np.concatenate([[x[0]], x[:-1]])
        x = 0.93 * x + 0.07 * prev2

    # 低出力段轻微再抬底
    if not np.isnan(cap) and cap > 0:
        r = x / cap
        r = np.where(r < 0.06, r + 0.0015 * (1.0 - r / 0.06), r)
        x = r * cap

    # 日历/日内时段微调（M11 轻量版）
    if not np.isnan(cap) and cap > 0 and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        r = x / cap
        mid = (hour >= 10.0) & (hour <= 14.0)
        r = np.where(mid & (r > 0.4), r * 0.995, r)
        x = r * cap

    # === 新增：清晨/傍晚低出力时段轻微上抬（针对 frac_below_02cap 高占比） ===
    if not np.isnan(cap) and cap > 0 and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        r = x / cap
        # 日出前后 6-9 点、日落前后 17-20 点，低出力段易低估，微抬
        edge = ((hour >= 6.0) & (hour <= 9.0)) | ((hour >= 17.0) & (hour <= 20.0))
        r = np.where(edge & (r > 0.02) & (r < 0.25), r * 1.006, r)
        x = r * cap

    # === 新增：高阶出力段轻微下压（p99 段易高估） ===
    if not np.isnan(cap) and cap > 0:
        r = x / cap
        r = np.where(r > 0.7, r - 0.002 * (r - 0.7) / 0.3, r)
        x = r * cap

    # M07 容量约束
    if not np.isnan(cap) and cap > 0:
        x = np.clip(x, 0.0, cap)

    # 防止常数序列
    if np.allclose(x, x[0]):
        x = x + np.linspace(0, 1e-6, n)

    return x