import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float)
    month = view['month'].values.astype(float)

    out = pp.copy()
    n = len(out)
    if n == 0:
        return out

    if np.isfinite(cap) and cap > 0:
        ratio = pp / cap
    else:
        ratio = pp / (np.nanmax(pp) + 1e-6)

    # 1) 低出力段夜间平滑
    low_mask = (ratio < 0.15) & ((hour < 6) | (hour >= 22))
    if np.any(low_mask):
        sm = pp.copy()
        for i in range(1, n - 1):
            if low_mask[i]:
                sm[i] = np.median(pp[i-1:i+2])
        out = np.where(low_mask, sm, out)

    # 2) 中高出力段爬坡异常限幅
    if n > 1:
        diff = np.diff(pp, prepend=pp[0])
        ramp_scale = np.nanpercentile(np.abs(diff), 95) + 1e-6
        mid_high = ratio > 0.30
        big_jump = np.abs(diff) > 3.0 * ramp_scale
        adjust_mask = mid_high & big_jump
        if np.any(adjust_mask):
            clipped = pp.copy()
            for i in range(n):
                if adjust_mask[i]:
                    prev = pp[i-1] if i > 0 else pp[i]
                    nxt = pp[i+1] if i < n-1 else pp[i]
                    clipped[i] = 0.5 * pp[i] + 0.5 * 0.5 * (prev + nxt)
            out = np.where(adjust_mask, clipped, out)

    # 3) 极高出力段向容量收缩
    if np.isfinite(cap) and cap > 0:
        very_high = ratio > 0.85
        if np.any(very_high):
            capped = np.minimum(out, 0.98 * cap)
            out = np.where(very_high, 0.7 * out + 0.3 * capped, out)

    # 4) 白天中低出力段轻度上修
    day_mask = (hour >= 8) & (hour <= 16) & (ratio > 0.1) & (ratio < 0.5)
    if np.any(day_mask) and n > 2:
        up = out.copy()
        for i in range(1, n - 1):
            if day_mask[i]:
                nb = 0.5 * (pp[i-1] + pp[i+1])
                if nb > pp[i]:
                    up[i] = pp[i] + 0.25 * (nb - pp[i])
        out = np.where(day_mask, up, out)

    # 5) 中低出力段孤立尖峰抑制
    if n > 2:
        mid_low = (ratio >= 0.15) & (ratio < 0.35) & (hour >= 7) & (hour <= 18)
        if np.any(mid_low):
            spike = out.copy()
            for i in range(1, n - 1):
                if mid_low[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    if out[i] > nb_mean * 1.4 and out[i] - nb_mean > 0.05 * cap:
                        spike[i] = 0.6 * out[i] + 0.4 * nb_mean
            out = np.where(mid_low, spike, out)

    # 6) 日出/日落过渡段平滑
    if n > 2:
        trans_mask = (((hour >= 5) & (hour <= 8)) | ((hour >= 17) & (hour <= 20))) & \
                     (ratio >= 0.05) & (ratio <= 0.25)
        if np.any(trans_mask):
            sm2 = out.copy()
            for i in range(1, n - 1):
                if trans_mask[i]:
                    sm2[i] = 0.6 * out[i] + 0.2 * out[i-1] + 0.2 * out[i+1]
            out = np.where(trans_mask, sm2, out)

    # 7) 中高出力段孤立低谷回填
    if n > 2:
        mid_band = (ratio >= 0.35) & (ratio <= 0.70) & (hour >= 7) & (hour <= 18)
        if np.any(mid_band):
            fill = out.copy()
            for i in range(1, n - 1):
                if mid_band[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    if out[i] < nb_mean * 0.7 and nb_mean - out[i] > 0.06 * cap:
                        fill[i] = 0.65 * out[i] + 0.35 * nb_mean
            out = np.where(mid_band, fill, out)

    # 8) 整体轻度向邻域均值收缩
    if n > 2:
        mid_all = (ratio > 0.10) & (ratio < 0.80)
        if np.any(mid_all):
            smooth = out.copy()
            for i in range(1, n - 1):
                if mid_all[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    smooth[i] = 0.85 * out[i] + 0.15 * nb_mean
            out = np.where(mid_all, smooth, out)

    # 9) 极低出力段(0<ratio<0.05)零值吸附
    if np.isfinite(cap) and cap > 0 and n > 2:
        tiny = (ratio > 0.0) & (ratio < 0.05)
        if np.any(tiny):
            shrink = out.copy()
            for i in range(1, n - 1):
                if tiny[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    if nb_mean < 0.05 * cap:
                        shrink[i] = 0.5 * out[i]
            out = np.where(tiny, shrink, out)

    # 10) 夜间低出力段(ratio<0.10)整体向下吸附
    if np.isfinite(cap) and cap > 0:
        night_low = ((hour < 5) | (hour >= 23)) & (ratio > 0.0) & (ratio < 0.10)
        if np.any(night_low):
            out = np.where(night_low, 0.85 * out, out)

    # 11) 中出力段(0.15~0.35)向邻域强平滑
    if n > 2:
        mid_noisy = (ratio >= 0.15) & (ratio < 0.35) & (hour >= 6) & (hour <= 19)
        if np.any(mid_noisy):
            sm3 = out.copy()
            for i in range(1, n - 1):
                if mid_noisy[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    sm3[i] = 0.75 * out[i] + 0.25 * nb_mean
            out = np.where(mid_noisy, sm3, out)

    # 12) 极低出力段(0<ratio<0.03)进一步向零吸附
    if np.isfinite(cap) and cap > 0:
        near_zero = (ratio > 0.0) & (ratio < 0.03)
        if np.any(near_zero):
            out = np.where(near_zero, 0.6 * out, out)

    # 13) 夜间(0-5, 22-24)中低出力段(0.10~0.20)轻度下修
    if np.isfinite(cap) and cap > 0:
        night_mid = ((hour < 5) | (hour >= 22)) & (ratio >= 0.10) & (ratio < 0.20)
        if np.any(night_mid):
            out = np.where(night_mid, 0.92 * out, out)

    # 14) 日出后(6-9点)低出力段(0.05~0.15)轻度上修
    if np.isfinite(cap) and cap > 0:
        sunrise = (hour >= 6) & (hour <= 9) & (ratio >= 0.05) & (ratio < 0.15)
        if np.any(sunrise):
            out = np.where(sunrise, 1.05 * out, out)

    # 15) 中高出力段(0.55~0.85)白天轻度向邻域收缩
    if n > 2:
        high_band = (ratio >= 0.55) & (ratio <= 0.85) & (hour >= 8) & (hour <= 17)
        if np.any(high_band):
            hs = out.copy()
            for i in range(1, n - 1):
                if high_band[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    hs[i] = 0.88 * out[i] + 0.12 * nb_mean
            out = np.where(high_band, hs, out)

    # 16) 午后(12-16点)中出力段(0.25~0.45)轻度下修
    if np.isfinite(cap) and cap > 0:
        pm_mid = (hour >= 12) & (hour <= 16) & (ratio >= 0.25) & (ratio < 0.45)
        if np.any(pm_mid):
            out = np.where(pm_mid, 0.96 * out, out)

    # 17) 极高出力段(ratio>0.90)白天向容量温和收缩
    if np.isfinite(cap) and cap > 0:
        extreme_high = (ratio > 0.90) & (hour >= 9) & (hour <= 16)
        if np.any(extreme_high):
            out = np.where(extreme_high, 0.95 * out, out)

    # 18) 中高出力段(0.45~0.60)白天轻度向邻域收缩
    if n > 2:
        mid_high_band = (ratio >= 0.45) & (ratio < 0.60) & (hour >= 9) & (hour <= 16)
        if np.any(mid_high_band):
            mh = out.copy()
            for i in range(1, n - 1):
                if mid_high_band[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    mh[i] = 0.90 * out[i] + 0.10 * nb_mean
            out = np.where(mid_high_band, mh, out)

    # 19) 夜间极低出力段(0<ratio<0.02)向零强吸附
    if np.isfinite(cap) and cap > 0:
        night_tiny = ((hour < 5) | (hour >= 22)) & (ratio > 0.0) & (ratio < 0.02)
        if np.any(night_tiny):
            out = np.where(night_tiny, 0.5 * out, out)

    # 20) 白天中出力段(0.35~0.45)轻度向邻域收缩
    if n > 2:
        mid_mid = (ratio >= 0.35) & (ratio < 0.45) & (hour >= 9) & (hour <= 16)
        if np.any(mid_mid):
            mm = out.copy()
            for i in range(1, n - 1):
                if mid_mid[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    mm[i] = 0.92 * out[i] + 0.08 * nb_mean
            out = np.where(mid_mid, mm, out)

    # 21) 凌晨(3-6点)低出力段(0.05~0.15)轻度下修
    if np.isfinite(cap) and cap > 0:
        pre_dawn = (hour >= 3) & (hour < 6) & (ratio >= 0.05) & (ratio < 0.15)
        if np.any(pre_dawn):
            out = np.where(pre_dawn, 0.94 * out, out)

    # 22) 夜间(0-4, 23-24)极低出力段(0.02~0.05)轻度下修
    if np.isfinite(cap) and cap > 0:
        night_verylow = ((hour < 4) | (hour >= 23)) & (ratio >= 0.02) & (ratio < 0.05)
        if np.any(night_verylow):
            out = np.where(night_verylow, 0.80 * out, out)

    # 23) 日出前(4-7点)极低出力段(0<ratio<0.03)向零吸附
    if np.isfinite(cap) and cap > 0:
        dawn_tiny = (hour >= 4) & (hour < 7) & (ratio > 0.0) & (ratio < 0.03)
        if np.any(dawn_tiny):
            out = np.where(dawn_tiny, 0.55 * out, out)

    # 24) 日落过渡(17-20点)低出力段(0.10~0.25)轻度下修
    if np.isfinite(cap) and cap > 0:
        sunset_mid = (hour >= 17) & (hour <= 20) & (ratio >= 0.10) & (ratio < 0.25)
        if np.any(sunset_mid):
            out = np.where(sunset_mid, 0.93 * out, out)

    # 25) 白天中高出力段(0.60~0.85)轻度向邻域收缩
    if n > 2:
        day_high = (ratio >= 0.60) & (ratio <= 0.85) & (hour >= 9) & (hour <= 16)
        if np.any(day_high):
            dh = out.copy()
            for i in range(1, n - 1):
                if day_high[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    dh[i] = 0.90 * out[i] + 0.10 * nb_mean
            out = np.where(day_high, dh, out)

    # 26) 夜间(21-24点)低出力段(0.03~0.08)轻度下修
    if np.isfinite(cap) and cap > 0:
        night_dusk_low = (hour >= 21) & (hour < 24) & (ratio >= 0.03) & (ratio < 0.08)
        if np.any(night_dusk_low):
            out = np.where(night_dusk_low, 0.88 * out, out)

    # 27) 清晨(5-7点)低出力段(0.03~0.08)轻度下修
    if np.isfinite(cap) and cap > 0:
        dawn_low = (hour >= 5) & (hour < 7) & (ratio >= 0.03) & (ratio < 0.08)
        if np.any(dawn_low):
            out = np.where(dawn_low, 0.90 * out, out)

    # 28) 新增：白天(10-15点)中高出力段(0.50~0.65)向邻域轻度收缩
    #     该段预测波动大，邻域平均更稳
    if n > 2:
        day_mid_high = (ratio >= 0.50) & (ratio < 0.65) & (hour >= 10) & (hour <= 15)
        if np.any(day_mid_high):
            dmh = out.copy()
            for i in range(1, n - 1):
                if day_mid_high[i]:
                    nb_mean = 0.5 * (out[i-1] + out[i+1])
                    dmh[i] = 0.92 * out[i] + 0.08 * nb_mean
            out = np.where(day_mid_high, dmh, out)

    out = np.maximum(out, 0.0)
    return out