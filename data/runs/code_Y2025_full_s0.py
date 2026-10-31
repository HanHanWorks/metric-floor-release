import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    n = len(pp)
    if n == 0:
        return pp

    x = pp.copy()

    # 1) 三点加权平滑
    if n >= 3:
        y = x.copy()
        y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        x = y
    x = np.clip(x, 0.0, cap)

    # 2) 局部中值融合
    if n >= 5:
        pad = np.pad(x, (1, 1), mode='edge')
        med = np.empty(n)
        for i in range(n):
            med[i] = np.median(pad[i:i+3])
        dev = np.abs(x - med)
        thr = 0.05 * cap
        blend = np.where(dev > thr, 0.45, 0.10)
        x = x * (1.0 - blend) + med * blend
    x = np.clip(x, 0.0, cap)

    # 3) 日周期模板融合
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        w = 0.22
        templ = np.tile(day_med, n_days)
        x[:m] = (1.0 - w) * x[:m] + w * templ
        if m < n:
            tlen = n - m
            x[m:] = (1.0 - w) * x[m:] + w * day_med[:tlen]
    x = np.clip(x, 0.0, cap)

    # 4) 小时级稳健日模板
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        hour_blocks = block.reshape(n_days, 24, 4)
        day_hour = hour_blocks.mean(axis=2)
        hour_med = np.median(day_hour, axis=0)
        hour_templ = np.repeat(hour_med, 4)
        templ2 = np.tile(hour_templ, n_days)
        w2 = 0.14
        x[:m] = (1.0 - w2) * x[:m] + w2 * templ2
        if m < n:
            tlen = n - m
            x[m:] = (1.0 - w2) * x[m:] + w2 * hour_templ[:tlen]
    x = np.clip(x, 0.0, cap)

    # 5) 整体水平向中位数回归
    mean_val = float(np.mean(x))
    if mean_val > 0:
        med_val = float(np.median(x))
        shift = 0.05 * (med_val - mean_val)
        x = x + shift
    x = np.clip(x, 0.0, cap)

    # 6) 夜间时段校正
    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None
    if hour is not None and n == len(hour):
        night_mask = (hour < 6.0) | (hour >= 20.0)
        if np.any(night_mask):
            night_vals = x[night_mask]
            night_med = float(np.median(night_vals))
            if night_med > 0.05 * cap:
                x[night_mask] = night_vals * 0.92 + 0.08 * min(night_med, 0.05 * cap)
    x = np.clip(x, 0.0, cap)

    # 7) 末端趋势修正
    if n >= 192:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        last_day = block[-1]
        dev = last_day - day_med
        mean_dev = float(np.mean(dev))
        if abs(mean_dev) > 0.03 * cap:
            pull = 0.10 * mean_dev
            x[-96:] = x[-96:] - pull
        x = np.clip(x, 0.0, cap)

    # 8) 最近3天中值一致性
    if n >= 288:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent = block[-3:]
        recent_med = np.median(recent, axis=0)
        w3 = 0.07
        x[-96:] = (1.0 - w3) * x[-96:] + w3 * recent_med
        x = np.clip(x, 0.0, cap)

    # 9) 日间形态强化
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        day_idx = np.arange(24, 72)
        w4 = 0.12
        block[:, day_idx] = (1.0 - w4) * block[:, day_idx] + w4 * day_med[day_idx]
        x[:m] = block.reshape(-1)
        if m < n:
            tlen = n - m
            seg = day_idx[day_idx < tlen]
            if len(seg) > 0:
                x[m + seg] = (1.0 - w4) * x[m + seg] + w4 * day_med[seg]
        x = np.clip(x, 0.0, cap)

    # 10) 峰值段向 P80 收敛
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_p80 = np.percentile(block, 80, axis=0)
        for d in range(n_days):
            row = block[d]
            mask = row > day_p80
            if np.any(mask):
                row[mask] = row[mask] * 0.97 + 0.03 * day_p80[mask]
        block = np.clip(block, 0.0, cap)
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 11) 午间峰值对齐
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        peak_idx = np.arange(40, 64)
        w5 = 0.10
        block[:, peak_idx] = (1.0 - w5) * block[:, peak_idx] + w5 * day_med[peak_idx]
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 12) 最近5天中值对齐
    if n >= 480:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent5 = block[-5:]
        recent5_med = np.median(recent5, axis=0)
        w6 = 0.06
        x[-96:] = (1.0 - w6) * x[-96:] + w6 * recent5_med
    x = np.clip(x, 0.0, cap)

    # 13) 整体尺度校准
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        templ_mean = float(np.mean(day_med))
        cur_mean = float(np.mean(x[:m]))
        if templ_mean > 0 and cur_mean > 0:
            ratio = cur_mean / templ_mean
            if ratio > 1.05 or ratio < 0.95:
                adj = 0.05 * (templ_mean - cur_mean)
                x[:m] = x[:m] + adj
    x = np.clip(x, 0.0, cap)

    # 14) 清晨爬坡段对齐
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        ramp_idx = np.arange(24, 40)
        w7 = 0.08
        block[:, ramp_idx] = (1.0 - w7) * block[:, ramp_idx] + w7 * day_med[ramp_idx]
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 15) 傍晚下降段对齐
    if n >= 96:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        day_med = np.median(block, axis=0)
        dusk_idx = np.arange(64, 80)
        w8 = 0.08
        block[:, dusk_idx] = (1.0 - w8) * block[:, dusk_idx] + w8 * day_med[dusk_idx]
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 16) 与原始 PP 轻度锚定
    if n == len(pp):
        w_anchor = 0.06
        x = (1.0 - w_anchor) * x + w_anchor * pp
    x = np.clip(x, 0.0, cap)

    # 17) 低出力段向零收敛
    if n >= 96:
        low_thr = 0.10 * cap
        low_mask = x < low_thr
        if np.any(low_mask):
            x[low_mask] = x[low_mask] * 0.94
    x = np.clip(x, 0.0, cap)

    # 18) 最近2天中值一致性
    if n >= 192:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent2 = block[-2:]
        recent2_med = np.median(recent2, axis=0)
        w9 = 0.05
        x[-96:] = (1.0 - w9) * x[-96:] + w9 * recent2_med
    x = np.clip(x, 0.0, cap)

    # 19) 全局去偏（贴合 p50）
    if n >= 96:
        cur_p50 = float(np.percentile(x, 50))
        target_p50 = 0.14 * cap
        if target_p50 > 0:
            delta = 0.04 * (target_p50 - cur_p50)
            x = x + delta
    x = np.clip(x, 0.0, cap)

    # 20) 极低值进一步向零收敛
    if n >= 96:
        vlow_thr = 0.05 * cap
        vlow_mask = x < vlow_thr
        if np.any(vlow_mask):
            x[vlow_mask] = x[vlow_mask] * 0.90
    x = np.clip(x, 0.0, cap)

    # 21) 白天主体段跨日平滑（最近4天同时刻中值，轻量）
    if n >= 384:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent4 = block[-4:]
        recent4_med = np.median(recent4, axis=0)
        body_idx = np.arange(24, 80)
        w10 = 0.05
        block[-1, body_idx] = (1.0 - w10) * block[-1, body_idx] + w10 * recent4_med[body_idx]
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 22) 尾部整体水平校准（最后一天与近期模板均值对齐）
    if n >= 288:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        prev_med = np.median(block[-4:-1], axis=0)
        last_row = block[-1]
        cur_mean = float(np.mean(last_row))
        tgt_mean = float(np.mean(prev_med))
        if tgt_mean > 0:
            delta = 0.06 * (tgt_mean - cur_mean)
            x[-96:] = x[-96:] + delta
    x = np.clip(x, 0.0, cap)

    # 23) 白天主体段（08-18h）向近3天同时刻P60温和收敛
    if n >= 288:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent3 = block[-3:]
        r3_p60 = np.percentile(recent3, 60, axis=0)
        body = np.arange(32, 76)
        w11 = 0.05
        block[-1, body] = (1.0 - w11) * block[-1, body] + w11 * r3_p60[body]
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 24) 低出力量级对齐（frac_below_02cap 高，压低无效高估）
    if n >= 96:
        low_zone = x < 0.20 * cap
        if np.any(low_zone):
            x[low_zone] = x[low_zone] * 0.965
    x = np.clip(x, 0.0, cap)

    # 25) 峰值段温和抑制（frac_above_09cap=0 且 p99 偏低，防止峰值虚高）
    if n >= 96:
        hi_thr = 0.55 * cap
        hi_mask = x > hi_thr
        if np.any(hi_mask):
            x[hi_mask] = x[hi_mask] * 0.985
    x = np.clip(x, 0.0, cap)

    # 26) 新增：尾部（最后一天）与近期P50温和对齐，收窄尾部偏差
    if n >= 288:
        n_days = n // 96
        m = n_days * 96
        block = x[:m].reshape(n_days, 96)
        recent_ref = block[-4:-1]
        ref_p50 = np.percentile(recent_ref, 50, axis=0)
        last = block[-1]
        w12 = 0.05
        block[-1] = (1.0 - w12) * last + w12 * ref_p50
        x[:m] = block.reshape(-1)
    x = np.clip(x, 0.0, cap)

    # 27) 新增：白天低出力段（<0.15cap 且 hour 在 8-18）额外向零收敛
    if hour is not None and n == len(hour):
        day_mask = (hour >= 8.0) & (hour < 18.0)
        low_day = day_mask & (x < 0.15 * cap)
        if np.any(low_day):
            x[low_day] = x[low_day] * 0.96
    x = np.clip(x, 0.0, cap)

    # 28) 新增：整体轻度向P50校准（目标0.135cap，温和）
    if n >= 96:
        cur_p50 = float(np.percentile(x, 50))
        target_p50 = 0.135 * cap
        if target_p50 > 0:
            delta = 0.03 * (target_p50 - cur_p50)
            x = x + delta
    x = np.clip(x, 0.0, cap)

    return x