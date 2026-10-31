import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    n = len(pp)
    if n == 0:
        return pp

    # 1) 中值滤波去毛刺（M03）
    out = pp.copy()
    k = 3
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        out[i] = np.median(pp[lo:hi])

    # 2) 滞后混合（M09）
    mixed = out.copy()
    alpha = 0.15
    for i in range(1, n):
        mixed[i] = (1 - alpha) * out[i] + alpha * out[i - 1]
    out = mixed

    # 3) 低出力收缩（M06）
    low_thr = 0.10 * cap
    for i in range(n):
        if out[i] < low_thr:
            out[i] = 0.85 * out[i] + 0.15 * low_thr

    # 4) 抬底（M05）
    floor = 0.005 * cap
    out = np.maximum(out, floor)

    # 5) 容量约束（M07）
    out = np.clip(out, 0.0, cap)

    # 6) 爬坡限制（M08）
    max_ramp = 0.06 * cap
    for i in range(1, n):
        d = out[i] - out[i - 1]
        if d > max_ramp:
            out[i] = out[i - 1] + max_ramp
        elif d < -max_ramp:
            out[i] = out[i - 1] - max_ramp

    # 7) 轻度全局缩放（M01 变体）
    out = out * 1.01
    out = np.clip(out, 0.0, cap)

    # 8) 分段映射（M12）
    for i in range(n):
        v = out[i]
        r = v / cap if cap > 0 else 0.0
        if r < 0.05:
            out[i] = v + 0.004 * cap
        elif r < 0.30:
            out[i] = v * 1.03
        elif r < 0.60:
            out[i] = v * 1.015
        else:
            out[i] = v * 0.995
    out = np.clip(out, 0.0, cap)

    # 9) 二次轻度平滑（M02 变体）
    smooth = out.copy()
    beta = 0.08
    for i in range(1, n - 1):
        smooth[i] = (1 - beta) * out[i] + beta * 0.5 * (out[i - 1] + out[i + 1])
    out = smooth
    out = np.clip(out, 0.0, cap)

    # 10) 日周期一致性校正（高斯加权低通）
    if n >= 96:
        day = out.copy()
        win = 5
        for i in range(n):
            lo = max(0, i - win)
            hi = min(n, i + win + 1)
            seg = out[lo:hi]
            w = np.exp(-0.5 * ((np.arange(lo, hi) - i) / (win * 0.6)) ** 2)
            day[i] = np.sum(seg * w) / np.sum(w)
        out = 0.75 * out + 0.25 * day
        out = np.clip(out, 0.0, cap)

    # 11) 分位数温和拉伸（M10 变体）
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.30 <= r <= 0.70:
            target = 0.5 * cap
            out[i] = 0.97 * out[i] + 0.03 * target
    out = np.clip(out, 0.0, cap)

    # 12) 日周期形状对齐（hour 均值靠拢）
    if 'hour' in view.columns and n >= 96:
        hour = view['hour'].values.astype(float)
        hour_bins = np.floor(hour).astype(int) % 24
        hour_mean = np.zeros(24)
        hour_cnt = np.zeros(24)
        for i in range(n):
            h = hour_bins[i]
            hour_mean[h] += out[i]
            hour_cnt[h] += 1
        for h in range(24):
            if hour_cnt[h] > 0:
                hour_mean[h] /= hour_cnt[h]
        gamma = 0.05
        for i in range(n):
            h = hour_bins[i]
            out[i] = (1 - gamma) * out[i] + gamma * hour_mean[h]
        out = np.clip(out, 0.0, cap)

    # 13) 爬坡方向一致性微调
    if n >= 3:
        adj = out.copy()
        for i in range(1, n - 1):
            d_prev = out[i] - out[i - 1]
            d_next = out[i + 1] - out[i]
            if d_prev * d_next < 0 and abs(d_prev) < 0.01 * cap and abs(d_next) < 0.01 * cap:
                adj[i] = 0.5 * (out[i - 1] + out[i + 1])
        out = adj
        out = np.clip(out, 0.0, cap)

    # 14) 日内形状对比度拉伸
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            m = np.mean(seg)
            if m > 1e-6:
                seg2 = m + 1.06 * (seg - m)
                seg2 = np.clip(seg2, 0.0, cap)
                out[s:e] = seg2
        rem = n % block
        if rem > 0:
            s = nblocks * block
            seg = out[s:]
            m = np.mean(seg)
            if m > 1e-6:
                seg2 = m + 1.06 * (seg - m)
                seg2 = np.clip(seg2, 0.0, cap)
                out[s:] = seg2
        out = np.clip(out, 0.0, cap)

    # 15) 基于 dow 的轻度周内一致性
    if 'dow' in view.columns and n >= 96:
        dow = view['dow'].values.astype(int) % 7
        dow_mean = np.zeros(7)
        dow_cnt = np.zeros(7)
        for i in range(n):
            d = dow[i]
            dow_mean[d] += out[i]
            dow_cnt[d] += 1
        for d in range(7):
            if dow_cnt[d] > 0:
                dow_mean[d] /= dow_cnt[d]
        delta = 0.02
        for i in range(n):
            d = dow[i]
            out[i] = (1 - delta) * out[i] + delta * dow_mean[d]
        out = np.clip(out, 0.0, cap)

    # 16) 峰段轻度抬升
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.25 <= r <= 0.55:
            out[i] = out[i] * 1.008
    out = np.clip(out, 0.0, cap)

    # 17) 极低出力段压向 0
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if r < 0.02:
            out[i] = out[i] * 0.85
    out = np.clip(out, 0.0, cap)

    # 18) 日块内极值温和回归
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            mx = np.max(seg)
            mn = np.min(seg)
            if mx - mn > 1e-6:
                m = np.mean(seg)
                idx_max = np.argmax(seg)
                idx_min = np.argmin(seg)
                seg[idx_max] = 0.97 * seg[idx_max] + 0.03 * m
                seg[idx_min] = 0.97 * seg[idx_min] + 0.03 * m
                out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 19) 低出力段（0.02~0.10cap）温和下压
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.02 <= r < 0.10:
            out[i] = out[i] * 0.985
    out = np.clip(out, 0.0, cap)

    # 20) 中段（0.10~0.22cap）轻微抬升
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.10 <= r < 0.22:
            out[i] = out[i] * 1.006
    out = np.clip(out, 0.0, cap)

    # 21) 跨日边界平滑
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(1, nblocks):
            idx = b * block
            if 0 < idx < n:
                out[idx] = 0.85 * out[idx] + 0.15 * out[idx - 1]
        out = np.clip(out, 0.0, cap)

    # 22) 极低段（<0.015cap）向零收缩
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if r < 0.015:
            out[i] = out[i] * 0.92
    out = np.clip(out, 0.0, cap)

    # 23) 夜间/低出力时段轻微下压
    if 'hour' in view.columns and n >= 1:
        hour = view['hour'].values.astype(float)
        for i in range(n):
            h = hour[i]
            r = out[i] / cap if cap > 0 else 0.0
            if (h < 6.0 or h >= 21.0) and r < 0.08:
                out[i] = out[i] * 0.97
        out = np.clip(out, 0.0, cap)

    # 24) 日块内低分位温和下压
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            thr = 0.05 * cap
            for j in range(len(seg)):
                if seg[j] < thr:
                    seg[j] = seg[j] * 0.97
            out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 25) 中高段（0.55~0.75cap）轻微下压
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.55 <= r <= 0.75:
            out[i] = out[i] * 0.994
    out = np.clip(out, 0.0, cap)

    # 26) 日块内低段整体温和下压
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            for j in range(len(seg)):
                r = seg[j] / cap if cap > 0 else 0.0
                if 0.05 <= r < 0.15:
                    seg[j] = seg[j] * 0.993
            out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 27) 日块内峰段（>0.35cap）轻微抬升
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            for j in range(len(seg)):
                r = seg[j] / cap if cap > 0 else 0.0
                if 0.35 <= r <= 0.55:
                    seg[j] = seg[j] * 1.006
            out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 28) 整体轻微上偏校正
    out = out * 1.005
    out = np.clip(out, 0.0, cap)

    # 29) 低出力主导段（0.15~0.25cap）温和抬升，改善 ACC 命中
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.15 <= r < 0.25:
            out[i] = out[i] * 1.004
    out = np.clip(out, 0.0, cap)

    # 30) 极低段（0.02~0.05cap）进一步轻微下压
    for i in range(n):
        r = out[i] / cap if cap > 0 else 0.0
        if 0.02 <= r < 0.05:
            out[i] = out[i] * 0.992
    out = np.clip(out, 0.0, cap)

    # 31) 日块内中段（0.20~0.35cap）轻微抬升
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            for j in range(len(seg)):
                r = seg[j] / cap if cap > 0 else 0.0
                if 0.20 <= r < 0.35:
                    seg[j] = seg[j] * 1.004
            out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 32) 基于 hour 的午间轻微抬升（光伏出力高峰时段）
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        for i in range(n):
            h = hour[i]
            r = out[i] / cap if cap > 0 else 0.0
            if 10.0 <= h < 14.0 and 0.20 <= r <= 0.50:
                out[i] = out[i] * 1.003
        out = np.clip(out, 0.0, cap)

    # 33) 新增：基于 hour 的清晨/傍晚过渡段轻微下压（出力爬升/下降期不确定性大）
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        for i in range(n):
            h = hour[i]
            r = out[i] / cap if cap > 0 else 0.0
            if (6.0 <= h < 8.5 or 16.5 <= h < 19.0) and 0.02 <= r < 0.15:
                out[i] = out[i] * 0.995
        out = np.clip(out, 0.0, cap)

    # 34) 新增：日块内中高段（0.25~0.45cap）轻微抬升，强化主出力区间
    if n >= 96:
        block = 96
        nblocks = n // block
        for b in range(nblocks):
            s = b * block
            e = s + block
            seg = out[s:e]
            for j in range(len(seg)):
                r = seg[j] / cap if cap > 0 else 0.0
                if 0.25 <= r < 0.45:
                    seg[j] = seg[j] * 1.003
            out[s:e] = seg
        out = np.clip(out, 0.0, cap)

    # 35) 新增：整体极轻微上偏（ACC 对小幅欠预测更敏感）
    out = out * 1.002
    out = np.clip(out, 0.0, cap)

    return out