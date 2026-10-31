import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 0.0
    n = len(pp)
    if n == 0:
        return pp
    if cap <= 0:
        return pp.copy()

    x = pp.copy()
    day_len = 96
    hour = view['hour'].values.astype(float)

    # 1) 轻度三点平滑（端点保留）
    if n >= 3:
        kernel = np.array([0.2, 0.6, 0.2])
        sm = np.convolve(x, kernel, mode='same')
        sm[0] = x[0]
        sm[-1] = x[-1]
        x = sm

    # 2) 容量约束
    x = np.clip(x, 0.0, cap)

    # 3) 低出力温和收缩
    low_mask = x < 0.15 * cap
    x = np.where(low_mask, x * 0.90, x)

    # 4) 极低出力归零
    tiny = x < 0.006 * cap
    x = np.where(tiny, 0.0, x)

    # 5) 夜间收缩
    night_mask = (hour < 6.0) | (hour >= 20.0)
    x = np.where(night_mask, x * 0.85, x)

    # 6) 正午高峰轻提
    noon_mask = (hour >= 10.0) & (hour <= 15.0) & (x > 0.35 * cap)
    x = np.where(noon_mask, x * 1.02, x)

    # 7) 过渡段抑制
    dawn_mask = (hour >= 6.0) & (hour < 8.0)
    dusk_mask = (hour >= 17.0) & (hour < 20.0)
    transition_mask = dawn_mask | dusk_mask
    x = np.where(transition_mask, x * 0.93, x)

    # 8) 按日水平校准
    if n >= day_len:
        n_days = n // day_len
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e]
            ratio = float(np.mean(seg)) / cap
            if ratio > 0.25:
                x[s:e] = seg * 0.97
            elif ratio < 0.08:
                x[s:e] = seg * 1.03

    # 9) 极低段压缩
    very_low = x < 0.05 * cap
    x = np.where(very_low, x * 0.85, x)

    # 10) 中低段轻收缩
    mid_low = (x >= 0.10 * cap) & (x < 0.30 * cap)
    x = np.where(mid_low, x * 0.96, x)

    # 11) 高峰保持
    high_mask = x > 0.60 * cap
    x = np.where(high_mask, x * 1.01, x)

    # 12) 全局温和收缩
    global_mid = (x >= 0.05 * cap) & (x < 0.60 * cap)
    x = np.where(global_mid, x * 0.985, x)

    # 13) 日内形状归一化
    if n >= day_len:
        n_days = n // day_len
        target_ratio = 0.17
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e]
            seg_mean = float(np.mean(seg))
            if seg_mean <= 0:
                continue
            ratio = seg_mean / cap
            if ratio > target_ratio * 1.6:
                x[s:e] = seg * 0.97
            elif ratio < target_ratio * 0.5:
                x[s:e] = seg * 1.02

    # 14) 按星期几偏差校正
    dow = view['dow'].values.astype(int)
    if n >= day_len:
        n_days = n // day_len
        dow_ratios = {}
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            dw = int(dow[s]) if s < n else 0
            seg_mean = float(np.mean(x[s:e]))
            dow_ratios.setdefault(dw, []).append(seg_mean / cap)
        dow_mean = {k: float(np.mean(v)) for k, v in dow_ratios.items() if len(v) > 0}
        overall = float(np.mean(list(dow_mean.values()))) if dow_mean else 0.17
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            dw = int(dow[s]) if s < n else 0
            if dw in dow_mean and overall > 0:
                adj = dow_mean[dw] / overall
                if adj > 1.08:
                    x[s:e] = x[s:e] * 0.98
                elif adj < 0.92:
                    x[s:e] = x[s:e] * 1.02

    # 15) 相邻日连续性
    if n >= 2 * day_len:
        n_days = n // day_len
        day_means = np.array([float(np.mean(x[d*day_len:(d+1)*day_len])) for d in range(n_days)])
        for d in range(n_days):
            if d == 0 or d == n_days - 1:
                continue
            neigh = 0.5 * (day_means[d-1] + day_means[d+1])
            if day_means[d] > 0 and neigh > 0:
                r = day_means[d] / neigh
                if r > 1.25:
                    x[d*day_len:(d+1)*day_len] *= 0.97
                elif r < 0.80:
                    x[d*day_len:(d+1)*day_len] *= 1.02

    # 16) 峰谷比校准
    if n >= day_len:
        n_days = n // day_len
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e]
            seg_max = float(np.max(seg))
            seg_mean = float(np.mean(seg))
            if seg_max <= 0 or seg_mean <= 0:
                continue
            peak_ratio = seg_max / cap
            if peak_ratio > 0.95:
                x[s:e] = seg * 0.98
            elif peak_ratio < 0.25 and seg_mean / cap > 0.05:
                x[s:e] = seg * 1.02

    # 17) hour 分组偏差校正
    if n >= day_len:
        hour_bins = np.clip(hour.astype(int), 0, 23)
        hour_sums = np.zeros(24)
        hour_cnts = np.zeros(24)
        for i in range(n):
            h = int(hour_bins[i])
            hour_sums[h] += x[i]
            hour_cnts[h] += 1
        hour_means = np.where(hour_cnts > 0, hour_sums / np.maximum(hour_cnts, 1), 0.0)
        for i in range(n):
            h = int(hour_bins[i])
            if hour_cnts[h] < 10:
                continue
            hm = hour_means[h]
            if hm <= 0:
                continue
            if x[i] > 1.5 * hm and x[i] > 0.02 * cap:
                x[i] *= 0.97
            elif x[i] < 0.5 * hm and x[i] > 0.01 * cap:
                x[i] *= 1.02

    # 18) 日间形态一致性
    if n >= 2 * day_len:
        n_days = n // day_len
        shapes = []
        means = []
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e]
            m = float(np.mean(seg))
            means.append(m)
            shapes.append(seg / m if m > 1e-6 else np.zeros(day_len))
        shapes = np.array(shapes)
        means = np.array(means)
        for d in range(n_days):
            if means[d] <= 1e-6:
                continue
            neigh_shapes = []
            if d > 0 and means[d-1] > 1e-6:
                neigh_shapes.append(shapes[d-1])
            if d < n_days - 1 and means[d+1] > 1e-6:
                neigh_shapes.append(shapes[d+1])
            if not neigh_shapes:
                continue
            ref_shape = np.mean(neigh_shapes, axis=0)
            if float(np.mean(np.abs(shapes[d] - ref_shape))) > 0.6:
                blend = 0.15
                new_shape = (1 - blend) * shapes[d] + blend * ref_shape
                s = d * day_len
                e = s + day_len
                x[s:e] = new_shape * means[d]

    # 19) 跨日同时刻稳健校准
    if n >= 3 * day_len:
        n_days = n // day_len
        hour_bins = np.clip(hour.astype(int), 0, 23)
        day_hour_mean = np.zeros((n_days, 24))
        day_hour_cnt = np.zeros((n_days, 24))
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            for i in range(s, e):
                h = int(hour_bins[i])
                day_hour_mean[d, h] += x[i]
                day_hour_cnt[d, h] += 1
        day_hour_mean = np.where(day_hour_cnt > 0, day_hour_mean / np.maximum(day_hour_cnt, 1), np.nan)
        hour_median = np.zeros(24)
        for h in range(24):
            col = day_hour_mean[:, h]
            col = col[~np.isnan(col)]
            hour_median[h] = float(np.median(col)) if len(col) >= 3 else np.nan
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            for i in range(s, e):
                h = int(hour_bins[i])
                hm = hour_median[h]
                if np.isnan(hm) or hm <= 0:
                    continue
                if x[i] > 1.8 * hm and x[i] > 0.05 * cap:
                    x[i] = 0.85 * x[i] + 0.15 * hm
                elif x[i] < 0.4 * hm and hm > 0.02 * cap and x[i] > 0.01 * cap:
                    x[i] = 0.92 * x[i] + 0.08 * hm

    # 20) 日内单调性修正
    if n >= day_len:
        n_days = n // day_len
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e].copy()
            for i in range(1, day_len - 1):
                h = hour[s + i]
                if 8.0 <= h <= 11.0:
                    if seg[i] < seg[i-1] * 0.7 and seg[i-1] > 0.1 * cap:
                        seg[i] = 0.7 * seg[i] + 0.3 * seg[i-1]
                elif 14.0 <= h <= 17.0:
                    if seg[i] > seg[i-1] * 1.4 and seg[i-1] > 0.1 * cap:
                        seg[i] = 0.7 * seg[i] + 0.3 * seg[i-1]
            x[s:e] = seg

    # 21) 光伏钟形形态对齐
    if n >= day_len:
        n_days = n // day_len
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e].copy()
            seg_max = float(np.max(seg))
            if seg_max <= 0.05 * cap:
                continue
            peak_idx = int(np.argmax(seg))
            peak_hour = hour[s + peak_idx]
            if peak_hour < 10.5 or peak_hour > 15.0:
                lo = max(0, peak_idx - 4)
                hi = min(day_len, peak_idx + 5)
                seg[lo:hi] = seg[lo:hi] * 0.985
            for i in range(day_len):
                h = hour[s + i]
                if (5.0 <= h < 7.0 or 18.0 <= h < 20.0) and seg[i] > 0.25 * seg_max:
                    seg[i] = seg[i] * 0.96
            x[s:e] = seg

    # 22) 跨日 hour 分位对齐
    if n >= 4 * day_len:
        hour_bins = np.clip(hour.astype(int), 0, 23)
        hour_vals = [[] for _ in range(24)]
        for i in range(n):
            hour_vals[int(hour_bins[i])].append(x[i])
        hour_p75 = np.full(24, np.nan)
        hour_p25 = np.full(24, np.nan)
        for h in range(24):
            if len(hour_vals[h]) >= 10:
                arr = np.array(hour_vals[h])
                hour_p75[h] = float(np.percentile(arr, 75))
                hour_p25[h] = float(np.percentile(arr, 25))
        for i in range(n):
            h = int(hour_bins[i])
            p75 = hour_p75[h]
            p25 = hour_p25[h]
            if np.isnan(p75) or p75 <= 0:
                continue
            if x[i] > 1.35 * p75 and x[i] > 0.08 * cap:
                x[i] = 0.90 * x[i] + 0.10 * p75
            elif (not np.isnan(p25)) and x[i] < 0.55 * p25 and x[i] > 0.01 * cap and p25 > 0.02 * cap:
                x[i] = 0.94 * x[i] + 0.06 * p25

    # 23) 新增：日内峰位与半高宽的钟形稳健性修正
    #     光伏曲线典型半高宽约 5-7 小时，对过宽/过窄形态轻微收敛
    if n >= day_len:
        n_days = n // day_len
        for d in range(n_days):
            s = d * day_len
            e = s + day_len
            seg = x[s:e].copy()
            seg_max = float(np.max(seg))
            if seg_max <= 0.10 * cap:
                continue
            half = 0.5 * seg_max
            above = seg >= half
            if not np.any(above):
                continue
            idx = np.where(above)[0]
            width_pts = int(idx[-1] - idx[0] + 1)
            width_hours = width_pts * 0.25
            # 典型半高宽 ~5.5h；过宽则轻微下压肩部，过窄则轻微抬升肩部
            if width_hours > 8.0:
                shoulder = (seg >= 0.3 * seg_max) & (seg < 0.7 * seg_max)
                seg = np.where(shoulder, seg * 0.985, seg)
            elif width_hours < 3.5:
                shoulder = (seg >= 0.3 * seg_max) & (seg < 0.7 * seg_max)
                seg = np.where(shoulder, seg * 1.015, seg)
            x[s:e] = seg

    # 24) 新增：整体分布向历史先验（pp_mean_ratio≈0.17）温和回归
    #     若整体均值明显偏离典型光伏场景，做小幅修正
    if n >= day_len:
        overall_ratio = float(np.mean(x)) / cap
        target = 0.17
        if overall_ratio > target * 1.35:
            x = x * 0.99
        elif overall_ratio < target * 0.65 and overall_ratio > 0:
            x = x * 1.01

    x = np.clip(x, 0.0, cap)
    return x