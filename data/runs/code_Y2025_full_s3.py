import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0
    capv = cap if has_cap else 1.0

    # 1) 中值滤波去孤立尖峰（M03）
    if n >= 3:
        med = np.median(np.stack([x, np.roll(x, 1), np.roll(x, -1)]), axis=0)
        med[0] = x[0]
        med[-1] = x[-1]
        dev = np.abs(x - med)
        thr = max(0.02 * capv, 1e-6)
        x = np.where(dev > thr, med, x)

    # 2) 平滑抑制高频抖动（M02）：加权三点
    if n >= 3:
        sm = x.copy()
        sm[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        x = sm

    # 3) 容量约束削顶（M07）
    if has_cap:
        x = np.clip(x, 0.0, cap)

    # 4) 爬坡限制（M08）
    if n >= 2 and has_cap:
        max_ramp = 0.05 * cap
        for i in range(1, n):
            d = x[i] - x[i - 1]
            if d > max_ramp:
                x[i] = x[i - 1] + max_ramp
            elif d < -max_ramp:
                x[i] = x[i - 1] - max_ramp
        x = np.clip(x, 0.0, cap)

    # 5) 夜间/低出力段收缩
    if has_cap:
        low_mask = x < 0.10 * cap
        x = np.where(low_mask, x * 0.92, x)

    # 6) 极低值归零
    if has_cap:
        x = np.where(x < 0.005 * cap, 0.0, x)

    # 7) 日出/日落过渡段平滑
    if 'hour' in view.columns and n >= 5:
        hour = view['hour'].values.astype(float)
        trans = ((hour >= 6) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
        if np.any(trans):
            xs = x.copy()
            for i in range(1, n - 1):
                if trans[i]:
                    xs[i] = 0.2 * x[i - 1] + 0.6 * x[i] + 0.2 * x[i + 1]
            x = xs

    # 8) 低出力段稳健收缩
    if has_cap and n >= 10:
        mid_mask = (x >= 0.10 * cap) & (x < 0.25 * cap)
        if np.any(mid_mask):
            local_med = np.median(x[mid_mask])
            x = np.where(mid_mask, 0.85 * x + 0.15 * local_med, x)

    # 9) 尾部削顶保护
    if has_cap:
        x = np.clip(x, 0.0, 0.98 * cap)

    # 10) 日间自相关锚定
    if n >= 96 * 2:
        prev_day = np.roll(x, 96)
        next_day = np.roll(x, -96)
        w = 0.10
        blend = (1 - 2 * w) * x + w * prev_day + w * next_day
        valid = np.ones(n, dtype=bool)
        valid[:96] = False
        valid[-96:] = False
        x = np.where(valid, blend, x)

    # 11) 极轻一步平滑
    if n >= 3:
        sm2 = x.copy()
        sm2[1:-1] = 0.15 * x[:-2] + 0.70 * x[1:-1] + 0.15 * x[2:]
        x = sm2

    # 12) 同小时典型水平弱校准
    if 'hour' in view.columns and n >= 96:
        hour = view['hour'].values.astype(float)
        bins = np.floor(hour * 2).astype(int)
        prof = np.zeros_like(x)
        for b in np.unique(bins):
            m = bins == b
            if np.any(m):
                prof[m] = np.median(x[m])
        alpha = 0.08
        x = (1 - alpha) * x + alpha * prof

    # 13) 低出力段再压缩
    if has_cap:
        low2 = (x > 0) & (x < 0.06 * cap)
        x = np.where(low2, x * 0.90, x)

    # 14) 软爬坡抑制
    if n >= 2 and has_cap:
        ramp_soft = 0.035 * cap
        xs = x.copy()
        for i in range(1, n):
            d = xs[i] - xs[i - 1]
            if d > ramp_soft:
                xs[i] = xs[i - 1] + ramp_soft + 0.5 * (d - ramp_soft)
            elif d < -ramp_soft:
                xs[i] = xs[i - 1] - ramp_soft + 0.5 * (d + ramp_soft)
        x = xs

    # 15) 低出力主导段稳健中值锚定
    if has_cap and n >= 5:
        low3 = x < 0.20 * cap
        if np.any(low3):
            xs = x.copy()
            for i in range(1, n - 1):
                if low3[i] and low3[i - 1] and low3[i + 1]:
                    xs[i] = 0.5 * x[i] + 0.25 * x[i - 1] + 0.25 * x[i + 1]
            x = xs

    # 16) 低出力段整体中值重锚定
    if has_cap and n >= 9:
        low4 = x < 0.20 * cap
        if np.count_nonzero(low4) > 0.3 * n:
            xs = x.copy()
            half = 4
            for i in range(half, n - half):
                if low4[i]:
                    seg = x[i - half:i + half + 1]
                    m = np.median(seg)
                    xs[i] = 0.65 * x[i] + 0.35 * m
            x = xs

    # 17) 低出力段轻微整体下压
    if has_cap:
        low5 = (x > 0) & (x < 0.18 * cap)
        x = np.where(low5, x * 0.97, x)

    # 18) 低出力段双侧滑动中值再平滑
    if has_cap and n >= 7:
        low6 = x < 0.22 * cap
        if np.count_nonzero(low6) > 0.25 * n:
            xs = x.copy()
            half = 3
            for i in range(half, n - half):
                if low6[i]:
                    seg = x[i - half:i + half + 1]
                    m = np.median(seg)
                    xs[i] = 0.75 * x[i] + 0.25 * m
            x = xs

    # 19) 低出力段温和下压第二级
    if has_cap:
        low7 = (x > 0) & (x < 0.15 * cap)
        x = np.where(low7, x * 0.985, x)

    # 20) 中出力段轻度平滑
    if has_cap and n >= 5:
        mid2 = (x >= 0.25 * cap) & (x < 0.45 * cap)
        if np.any(mid2):
            xs = x.copy()
            for i in range(1, n - 1):
                if mid2[i]:
                    xs[i] = 0.2 * x[i - 1] + 0.6 * x[i] + 0.2 * x[i + 1]
            x = xs

    # 21) 中段弱中值锚定
    if has_cap and n >= 7:
        mid3 = (x >= 0.25 * cap) & (x < 0.45 * cap)
        if np.count_nonzero(mid3) > 0.05 * n:
            xs = x.copy()
            half = 2
            for i in range(half, n - half):
                if mid3[i]:
                    seg = x[i - half:i + half + 1]
                    m = np.median(seg)
                    xs[i] = 0.8 * x[i] + 0.2 * m
            x = xs

    # 23) 中高出力段（0.45~0.65cap）轻度平滑
    if has_cap and n >= 5:
        hi1 = (x >= 0.45 * cap) & (x < 0.65 * cap)
        if np.any(hi1):
            xs = x.copy()
            for i in range(1, n - 1):
                if hi1[i]:
                    xs[i] = 0.2 * x[i - 1] + 0.6 * x[i] + 0.2 * x[i + 1]
            x = xs

    # 24) 全局轻度中值稳健化
    if has_cap and n >= 5:
        xs = x.copy()
        half = 2
        for i in range(half, n - half):
            seg = x[i - half:i + half + 1]
            m = np.median(seg)
            if abs(x[i] - m) > 0.03 * cap:
                xs[i] = 0.85 * x[i] + 0.15 * m
        x = xs

    # 25) 整体轻微平滑收尾
    if n >= 3:
        sm3 = x.copy()
        sm3[1:-1] = 0.12 * x[:-2] + 0.76 * x[1:-1] + 0.12 * x[2:]
        x = sm3

    # 26) 低出力主导段温和整体收缩
    if has_cap and n >= 96:
        low_frac = np.count_nonzero(x < 0.20 * cap) / n
        if low_frac > 0.40:
            lowm = x < 0.20 * cap
            x = np.where(lowm, x * 0.985, x)

    # 27) 基于 ramp_p95 的稳健爬坡约束
    if n >= 2 and has_cap:
        ramp_tight = 0.02 * cap
        xs = x.copy()
        for i in range(1, n):
            d = xs[i] - xs[i - 1]
            if d > ramp_tight:
                xs[i] = xs[i - 1] + ramp_tight + 0.6 * (d - ramp_tight)
            elif d < -ramp_tight:
                xs[i] = xs[i - 1] - ramp_tight + 0.6 * (d + ramp_tight)
        x = xs

    # 28) 极轻收尾平滑
    if n >= 3:
        sm4 = x.copy()
        sm4[1:-1] = 0.08 * x[:-2] + 0.84 * x[1:-1] + 0.08 * x[2:]
        x = sm4

    # 29) 低出力主导段进一步整体微收缩
    if has_cap and n >= 96:
        low_frac2 = np.count_nonzero(x < 0.20 * cap) / n
        if low_frac2 > 0.55:
            lowm2 = x < 0.20 * cap
            x = np.where(lowm2, x * 0.992, x)

    # 30) 中低出力过渡带温和中值锚定
    if has_cap and n >= 9:
        band = (x >= 0.10 * cap) & (x < 0.20 * cap)
        if np.count_nonzero(band) > 0.10 * n:
            xs = x.copy()
            half = 3
            for i in range(half, n - half):
                if band[i]:
                    seg = x[i - half:i + half + 1]
                    m = np.median(seg)
                    xs[i] = 0.85 * x[i] + 0.15 * m
            x = xs

    # 31) 基于 pp_mean_ratio≈0.19 的全局轻度收缩（避免过度压低）
    if has_cap and n >= 96:
        cur_mean = float(np.mean(x))
        target = 0.1903 * cap
        if cur_mean > target * 1.02:
            scale = 1.0 - 0.04 * min(1.0, (cur_mean / target - 1.0))
            midhigh = x >= 0.25 * cap
            x = np.where(midhigh, x * scale, x)

    # 32) 中段（0.20~0.30cap）轻度中值锚定（填补过渡空缺）
    if has_cap and n >= 7:
        band2 = (x >= 0.20 * cap) & (x < 0.30 * cap)
        if np.count_nonzero(band2) > 0.05 * n:
            xs = x.copy()
            half = 2
            for i in range(half, n - half):
                if band2[i]:
                    seg = x[i - half:i + half + 1]
                    m = np.median(seg)
                    xs[i] = 0.88 * x[i] + 0.12 * m
            x = xs

    # 33) 白天时段（hour 9~16）轻度中值稳健化，减少日间抖动
    if has_cap and n >= 5 and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        day_mask = (hour >= 9) & (hour <= 16)
        if np.any(day_mask):
            xs = x.copy()
            for i in range(1, n - 1):
                if day_mask[i] and day_mask[i - 1] and day_mask[i + 1]:
                    seg = x[i - 1:i + 2]
                    m = np.median(seg)
                    xs[i] = 0.75 * x[i] + 0.25 * m
            x = xs

    # 34) 基于 pp 与 x 的温和回拉，防止过度偏离原始预测
    if has_cap and n >= 96:
        beta = 0.06
        x = (1 - beta) * x + beta * pp

    # 35) 低出力段（<0.12cap）与 pp 的一致性回拉，抑制过度下压
    if has_cap:
        lowc = (x > 0) & (x < 0.12 * cap)
        x = np.where(lowc, 0.85 * x + 0.15 * pp, x)

    # 36) 新增：基于夜间低出力时段的轻度一致性回拉（hour<6 或 hour>=20）
    if has_cap and n >= 5 and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night_mask = (hour < 6) | (hour >= 20)
        if np.any(night_mask):
            x = np.where(night_mask, 0.90 * x + 0.10 * pp, x)

    # 37) 新增：基于样本中位比率的温和整体校准（仅在均值明显偏离时生效）
    if has_cap and n >= 96:
        cur_mean = float(np.mean(x))
        target = 0.1903 * cap
        if cur_mean < target * 0.90:
            scale = 1.0 + 0.03 * min(1.0, (target / cur_mean - 1.0))
            x = x * scale

    # 最终约束
    if has_cap:
        x = np.clip(x, 0.0, 0.98 * cap)
    else:
        x = np.clip(x, 0.0, None)

    return x