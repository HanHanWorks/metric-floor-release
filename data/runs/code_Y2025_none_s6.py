import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑（低功率段平滑更强，保留日周期形状）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = (out[:-2] + out[1:-1] + out[2:]) / 3.0
        if np.isfinite(cap):
            w = np.clip(1.0 - out / (cap + 1e-6), 0.0, 1.0)
        else:
            w = np.ones(n)
        out = w * sm + (1.0 - w) * out

    # 2) 夜间/低出力段向 0 收缩
    if 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        night = ((hour >= 22) | (hour < 6)).astype(float)
        if np.isfinite(cap):
            low = np.clip(1.0 - out / (0.15 * cap + 1e-6), 0.0, 1.0)
        else:
            low = np.clip(1.0 - out / (out.mean() + 1e-6), 0.0, 1.0)
        shrink = 0.35 * night * low
        out = out * (1.0 - shrink)

    # 3) 极端高值轻微压制
    if np.isfinite(cap):
        hi = out > 0.9 * cap
        if hi.any():
            out[hi] = 0.9 * cap + 0.5 * (out[hi] - 0.9 * cap)

    # 4) 非负裁剪
    out = np.clip(out, 0.0, None)

    # 5) 常数回退
    if np.allclose(out, out[0]):
        out = pp.copy()

    # 6) 向历史均值比轻微回归
    if np.isfinite(cap) and n > 0:
        mean_ratio = out.mean() / (cap + 1e-6)
        target_ratio = 0.2385
        if mean_ratio > target_ratio:
            scale = 1.0 - 0.06 * min(1.0, (mean_ratio - target_ratio) / (target_ratio + 1e-6))
            out = out * scale
        out = np.clip(out, 0.0, None)

    # 7) 低功率段收缩
    if np.isfinite(cap):
        low_mask = out < 0.20 * cap
        if low_mask.any():
            out[low_mask] = out[low_mask] * 0.97

    # 8) 日周期基线融合——对明显偏高的天做轻微下压
    if np.isfinite(cap) and n >= 96:
        n_full = (n // 96) * 96
        if n_full >= 96:
            day_means = out[:n_full].reshape(-1, 96).mean(axis=1) / (cap + 1e-6)
            target_day = 0.2385
            adj = np.ones(n_full // 96)
            over = day_means > target_day * 1.15
            if over.any():
                adj[over] = 1.0 - 0.03
            adj_full = np.repeat(adj, 96)
            out[:n_full] = out[:n_full] * adj_full
        out = np.clip(out, 0.0, None)

    # 9) ramp 尖峰抑制（温和限幅，保留形状）
    if n >= 3 and np.isfinite(cap):
        ramp_limit = 0.06 * cap
        diff = np.diff(out)
        large_up = diff > ramp_limit
        if large_up.any():
            idx = np.where(large_up)[0]
            for i in idx:
                excess = diff[i] - ramp_limit
                out[i + 1] -= 0.5 * excess
        out = np.clip(out, 0.0, None)

    # 10) 对超低出力段轻微上抬，避免过度压零
    if np.isfinite(cap):
        tiny = (out > 0) & (out < 0.05 * cap)
        if tiny.any():
            out[tiny] = out[tiny] * 1.02
        out = np.clip(out, 0.0, None)

    # 11) 向下 ramp 同样温和限幅
    if n >= 3 and np.isfinite(cap):
        ramp_limit = 0.06 * cap
        diff = np.diff(out)
        large_down = diff < -ramp_limit
        if large_down.any():
            idx = np.where(large_down)[0]
            for i in idx:
                excess = -diff[i] - ramp_limit
                out[i + 1] += 0.5 * excess
        out = np.clip(out, 0.0, None)

    # 12) 白天中高功率段的轻微收缩
    if np.isfinite(cap) and 'hour' in view.columns:
        hour = view['hour'].values.astype(float)
        day = ((hour >= 8) & (hour < 17)).astype(bool)
        mid_high = day & (out > 0.55 * cap)
        if mid_high.any():
            out[mid_high] = out[mid_high] * 0.985
        out = np.clip(out, 0.0, None)

    # 13) 相邻点间做极轻微二阶差分平滑（仅在低功率段）
    if n >= 3 and np.isfinite(cap):
        low_seg = out < 0.25 * cap
        if low_seg.any() and n >= 3:
            sm = out.copy()
            sm[1:-1] = (out[:-2] + 2.0 * out[1:-1] + out[2:]) / 4.0
            w = np.zeros(n)
            w[1:-1] = low_seg[1:-1].astype(float) * 0.15
            out = w * sm + (1.0 - w) * out
        out = np.clip(out, 0.0, None)

    # 14) 按 hour 学习日内典型形状，与日总量约束做轻度融合
    if np.isfinite(cap) and 'hour' in view.columns and n >= 96:
        hour = view['hour'].values.astype(float)
        slot = np.clip(np.floor(hour).astype(int), 0, 23)
        slot_mean = np.zeros(24)
        slot_cnt = np.zeros(24)
        for h in range(24):
            m = slot == h
            if m.any():
                slot_mean[h] = out[m].mean() / (cap + 1e-6)
                slot_cnt[h] = m.sum()
        if slot_mean.sum() > 0.5:
            shape = slot_mean / (slot_mean.mean() + 1e-6)
            cur_slot = slot_mean[slot]
            cur_slot = np.where(cur_slot > 0, cur_slot, slot_mean[slot])
            dev = out / (cap + 1e-6) - cur_slot
            big_dev = np.abs(dev) > 0.10
            if big_dev.any():
                target = cur_slot * cap
                out[big_dev] = 0.85 * out[big_dev] + 0.15 * target[big_dev]
            out = np.clip(out, 0.0, None)

    # 15) 整体均值向目标比温和校准（限制总能量偏差）
    if np.isfinite(cap) and n > 0:
        mean_ratio = out.mean() / (cap + 1e-6)
        target_ratio = 0.2385
        if abs(mean_ratio - target_ratio) > 0.08 * target_ratio and mean_ratio > 0:
            factor = target_ratio / mean_ratio
            factor = 1.0 + 0.5 * (factor - 1.0)
            factor = np.clip(factor, 0.92, 1.08)
            out = out * factor
        out = np.clip(out, 0.0, None)

    # 16) 新增：对明显偏高的日做更精细的日级缩放（按偏离程度分级）
    if np.isfinite(cap) and n >= 96:
        n_full = (n // 96) * 96
        if n_full >= 96:
            dm = out[:n_full].reshape(-1, 96).mean(axis=1) / (cap + 1e-6)
            target_day = 0.2385
            adj = np.ones(n_full // 96)
            # 分级处理：偏离越大，压下越多（但温和）
            for k, r in enumerate(dm):
                if r > target_day * 1.30:
                    adj[k] = 0.955
                elif r > target_day * 1.15:
                    adj[k] = 0.975
                elif r > target_day * 1.05:
                    adj[k] = 0.990
                elif r < target_day * 0.75 and r > 0:
                    adj[k] = 1.020
            out[:n_full] = out[:n_full] * np.repeat(adj, 96)
        out = np.clip(out, 0.0, None)

    # 17) 新增：白天峰值段轻微下压（高功率区间对考核误差更敏感）
    if np.isfinite(cap):
        peak = out > 0.70 * cap
        if peak.any():
            out[peak] = out[peak] * 0.99
        out = np.clip(out, 0.0, None)

    # 18) 常数回退（二次保障）
    if np.allclose(out, out[0]):
        out = pp.copy()

    return out