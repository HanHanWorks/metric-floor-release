import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 轻度中值滤波去尖峰（保留主要形态）
    if n >= 3:
        med = x.copy()
        med[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        x = 0.60 * x + 0.40 * med

    # 2) 轻度平滑（窄核）
    if n >= 5:
        k = np.array([1, 2, 3, 2, 1], dtype=float)
        k /= k.sum()
        xp = np.pad(x, 2, mode='edge')
        xs = np.convolve(xp, k, mode='valid')
        x = 0.68 * x + 0.32 * xs

    # 3) 低出力段轻微压缩
    ratio = x / max(cap, 1e-6)
    x = np.where(ratio < 0.10, x * 0.945, x)
    ratio2 = x / max(cap, 1e-6)
    mid_mask = (ratio2 >= 0.10) & (ratio2 < 0.25)
    x = np.where(mid_mask, x * 0.982, x)

    # 4) 极低出力轻微抬升
    ratio3 = x / max(cap, 1e-6)
    tiny_mask = (ratio3 > 0.0) & (ratio3 < 0.02)
    x = np.where(tiny_mask, x + 0.0030 * cap, x)

    x = np.clip(x, 0.0, cap)

    # 5) 基础爬坡限制（温和）
    max_ramp = 0.050 * cap
    if n >= 2:
        for i in range(1, n):
            d = x[i] - x[i - 1]
            if d > max_ramp:
                x[i] = x[i - 1] + max_ramp
            elif d < -max_ramp:
                x[i] = x[i - 1] - max_ramp

    # 6) 滞后混合（轻度）
    if n >= 2:
        y = x.copy()
        y[1:] = 0.90 * x[1:] + 0.10 * x[:-1]
        x = y

    # 7) 夜间轻微压低
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night_mask = (hour < 6) | (hour >= 21)
        x = np.where(night_mask, x * 0.980, x)

    # 8) 中高出力段轻微放大
    ratio4 = x / max(cap, 1e-6)
    high_mask = (ratio4 >= 0.35) & (ratio4 < 0.75)
    x = np.where(high_mask, x * 1.015, x)

    # 9) 日周期一致性：按天轻度中值锚定
    if n >= 96:
        days = n // 96
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            seg_med = np.median(seg)
            if seg_med > 1e-6:
                dev = seg - seg_med
                x[s:e] = seg_med + 0.935 * dev

    # 10) 日间一致性锚定（轻度）
    if n >= 192:
        days = n // 96
        for d in range(1, days):
            s = d * 96
            e = s + 96
            prev = x[s - 96:e - 96]
            cur = x[s:e]
            x[s:e] = 0.945 * cur + 0.055 * prev

    # 11) 分位数锚定
    if n >= 32:
        q_lo = np.percentile(x, 10)
        q_hi = np.percentile(x, 90)
        if q_hi - q_lo > 1e-6:
            x = np.where(x < q_lo, x * 0.970 + q_lo * 0.030, x)
            x = np.where(x > q_hi, x * 0.970 + q_hi * 0.030, x)

    # 12) 日间形状一致性
    if n >= 192:
        days = n // 96
        for d in range(1, days):
            s = d * 96
            e = s + 96
            prev = x[s - 96:e - 96]
            cur = x[s:e]
            pm = np.mean(prev)
            cm = np.mean(cur)
            if pm > 1e-6 and cm > 1e-6:
                prev_shape = prev / pm
                cur_shape = cur / cm
                blended_shape = 0.91 * cur_shape + 0.09 * prev_shape
                blended = blended_shape * cm
                x[s:e] = 0.90 * cur + 0.10 * blended

    # 13) 日内形状与典型出力曲线轻度对齐
    if n >= 96:
        days = n // 96
        slot_med = np.zeros(96)
        cnt = 0
        for d in range(days):
            s = d * 96
            e = s + 96
            slot_med += x[s:e]
            cnt += 1
        if cnt > 0:
            slot_med /= cnt
        sm = np.mean(slot_med)
        if sm > 1e-6:
            for d in range(days):
                s = d * 96
                e = s + 96
                cur = x[s:e]
                cm = np.mean(cur)
                if cm > 1e-6:
                    cur_shape = cur / cm
                    ref_shape = slot_med / sm
                    blended_shape = 0.92 * cur_shape + 0.08 * ref_shape
                    blended = blended_shape * cm
                    x[s:e] = 0.92 * cur + 0.08 * blended

    # 14) 跨天槽位中值形状稳健锚定
    if n >= 288:
        days = n // 96
        slot_stack = np.zeros((days, 96))
        for d in range(days):
            slot_stack[d] = x[d * 96:(d + 1) * 96]
        slot_med2 = np.median(slot_stack, axis=0)
        sm2 = np.mean(slot_med2)
        if sm2 > 1e-6:
            ref_shape2 = slot_med2 / sm2
            for d in range(days):
                s = d * 96
                e = s + 96
                cur = x[s:e]
                cm = np.mean(cur)
                if cm > 1e-6:
                    cur_shape = cur / cm
                    blended_shape = 0.94 * cur_shape + 0.06 * ref_shape2
                    blended = blended_shape * cm
                    x[s:e] = 0.94 * cur + 0.06 * blended

    # 15) 极低出力段软阈值
    if n >= 32:
        q05 = np.percentile(x, 5)
        if q05 > 1e-6:
            soft_mask = (x > 0) & (x < q05)
            x = np.where(soft_mask, x * 0.982, x)

    # 16) 高出力段轻微抑制
    if n >= 32:
        q95 = np.percentile(x, 95)
        if q95 > 1e-6:
            top_mask = x > q95
            x = np.where(top_mask, x * 0.985, x)

    # 17) 基于 ramp 分布的温和二次爬坡约束
    if n >= 3:
        diffs = np.abs(np.diff(x))
        if len(diffs) > 0:
            ramp_p95 = np.percentile(diffs, 95)
            ramp_cap = max(ramp_p95 * 2.0, 0.007 * cap)
            for i in range(1, n):
                d = x[i] - x[i - 1]
                if d > ramp_cap:
                    x[i] = x[i - 1] + ramp_cap
                elif d < -ramp_cap:
                    x[i] = x[i - 1] - ramp_cap

    # 18) 日内峰谷比温和拉伸
    if n >= 96:
        days = n // 96
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            m = np.mean(seg)
            if m > 1e-6:
                seg_norm = seg / m
                seg_stretch = m * (1.0 + 1.035 * (seg_norm - 1.0))
                x[s:e] = 0.935 * seg + 0.065 * seg_stretch

    # 19) 跨天槽位低分位稳健下限
    if n >= 192:
        days = n // 96
        slot_stack = np.zeros((days, 96))
        for d in range(days):
            slot_stack[d] = x[d * 96:(d + 1) * 96]
        slot_q20 = np.percentile(slot_stack, 20, axis=0)
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            floor_ref = slot_q20 * 0.60
            low_mask = seg < floor_ref
            seg = np.where(low_mask, seg * 0.96 + floor_ref * 0.04, seg)
            x[s:e] = seg

    # 20) 跨天槽位高分位温和压制
    if n >= 192:
        days = n // 96
        slot_stack = np.zeros((days, 96))
        for d in range(days):
            slot_stack[d] = x[d * 96:(d + 1) * 96]
        slot_q85 = np.percentile(slot_stack, 85, axis=0)
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            high_cap = slot_q85 * 1.06
            hi_mask = seg > high_cap
            seg = np.where(hi_mask, seg * 0.982 + high_cap * 0.018, seg)
            x[s:e] = seg

    # 21) 最终轻度滞后平滑
    if n >= 2:
        y = x.copy()
        y[1:] = 0.94 * x[1:] + 0.06 * x[:-1]
        x = y

    # 22) 基于日类型(dow)的温和校正
    if 'dow' in view.columns and n >= 96:
        dow = view['dow'].values.astype(int)
        days = n // 96
        weekend_slots = []
        weekday_slots = []
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            m = np.mean(seg)
            if m <= 1e-6:
                continue
            shape = seg / m
            if dow[s] >= 5:
                weekend_slots.append(shape)
            else:
                weekday_slots.append(shape)
        def _avg_shape(lst):
            if len(lst) == 0:
                return None
            arr = np.stack(lst, axis=0)
            sh = np.mean(arr, axis=0)
            sm = np.mean(sh)
            if sm > 1e-6:
                return sh / sm
            return None
        wk_shape = _avg_shape(weekday_slots)
        we_shape = _avg_shape(weekend_slots)
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            m = np.mean(seg)
            if m <= 1e-6:
                continue
            cur_shape = seg / m
            ref = we_shape if dow[s] >= 5 else wk_shape
            if ref is None:
                continue
            blended_shape = 0.955 * cur_shape + 0.045 * ref
            blended = blended_shape * m
            x[s:e] = 0.955 * seg + 0.045 * blended

    # 23) 时段分层轻度缩放
    if 'hour' in view.columns and n >= 32:
        hour = view['hour'].values.astype(float)
        morning = (hour >= 6) & (hour < 10)
        evening = (hour >= 16) & (hour < 20)
        noon = (hour >= 11) & (hour < 15)
        x = np.where(morning | evening, x * 1.008, x)
        x = np.where(noon, x * 0.994, x)

    # 24) 整体尺度轻度回缩
    if n >= 32:
        mean_ratio = np.mean(x) / max(cap, 1e-6)
        if mean_ratio > 0.30:
            x = x * 0.996

    # 25) 新：收尾段形状对齐（月度典型曲线轻度锚定）
    if 'month' in view.columns and n >= 192:
        month = view['month'].values.astype(int)
        days = n // 96
        # 按月收集槽位形状
        month_shapes = {}
        for d in range(days):
            s = d * 96
            e = s + 96
            seg = x[s:e]
            m = np.mean(seg)
            if m <= 1e-6:
                continue
            mo = month[s]
            month_shapes.setdefault(mo, []).append(seg / m)
        month_ref = {}
        for mo, lst in month_shapes.items():
            if len(lst) >= 2:
                arr = np.stack(lst, axis=0)
                sh = np.median(arr, axis=0)
                sm = np.mean(sh)
                if sm > 1e-6:
                    month_ref[mo] = sh / sm
        for d in range(days):
            s = d * 96
            e = s + 96
            mo = month[s]
            if mo not in month_ref:
                continue
            seg = x[s:e]
            m = np.mean(seg)
            if m <= 1e-6:
                continue
            cur_shape = seg / m
            ref = month_ref[mo]
            blended_shape = 0.975 * cur_shape + 0.025 * ref
            blended = blended_shape * m
            x[s:e] = 0.975 * seg + 0.025 * blended

    # 26) 新：尾部连续性增强——末点与前列趋势轻度对齐
    if n >= 8:
        tail_len = min(96, n)
        s = n - tail_len
        seg = x[s:n]
        m = np.mean(seg)
        if m > 1e-6:
            # 使用线性趋势残差轻微校正
            idx = np.arange(tail_len, dtype=float)
            coef = np.polyfit(idx, seg, 1)
            trend = np.polyval(coef, idx)
            resid = seg - trend
            # 残差轻度收敛
            x[s:n] = trend + 0.975 * resid

    x = np.clip(x, 0.0, cap)
    return x