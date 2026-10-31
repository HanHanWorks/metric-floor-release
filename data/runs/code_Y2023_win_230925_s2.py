import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    out = pp.copy()

    # 1) 轻度平滑：抑制高频抖动，保留日周期结构
    if n >= 5:
        med = pp.copy()
        med[1:-1] = np.median(np.vstack([pp[:-2], pp[1:-1], pp[2:]]), axis=0)
        ker = np.ones(5) / 5.0
        pad = np.pad(pp, 2, mode='edge')
        avg = np.convolve(pad, ker, mode='valid')
        smooth = 0.5 * med + 0.5 * avg
    else:
        smooth = pp.copy()

    hour = view['hour'].values.astype(float) if 'hour' in view.columns else None

    if hour is not None:
        night_w = 0.5 * (1.0 + np.cos(2.0 * np.pi * (hour - 0.0) / 24.0))
        day_w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour - 12.0) / 24.0))
        noon_w = day_w
        dusk_w = 0.5 * (1.0 + np.cos(2.0 * np.pi * (hour - 18.0) / 24.0))
    else:
        night_w = np.zeros(n)
        day_w = np.zeros(n)
        noon_w = np.zeros(n)
        dusk_w = np.zeros(n)

    # 2) 夜间低出力修正
    if hour is not None:
        if not np.isnan(cap) and cap > 0:
            low_mask = np.clip(1.0 - smooth / (0.35 * cap), 0.0, 1.0)
        else:
            low_mask = np.clip(1.0 - smooth / (np.nanmax(smooth) + 1e-6), 0.0, 1.0)
        shrink = 0.06 * night_w * low_mask
        out = smooth * (1.0 - shrink)
    else:
        out = smooth

    if not np.isnan(cap) and cap > 0:
        # 3) 低出力段整体轻微下压
        ratio = out / cap
        low_zone = np.clip((0.15 - ratio) / 0.15, 0.0, 1.0)
        shrink2 = 0.03 * low_zone * (0.4 + 0.6 * (1.0 - day_w))
        out = out * (1.0 - shrink2)

        # 3b) 极低出力段进一步温和下压
        ratio2 = out / cap
        very_low = np.clip((0.06 - ratio2) / 0.06, 0.0, 1.0)
        shrink3 = 0.025 * very_low * (0.5 + 0.5 * night_w)
        out = out * (1.0 - shrink3)

        # 3c) 中高出力段轻微回调
        ratio3 = out / cap
        high_zone = np.clip((ratio3 - 0.5) / 0.5, 0.0, 1.0)
        shrink4 = 0.012 * high_zone * (1.0 - 0.5 * noon_w)
        out = out * (1.0 - shrink4)

        # 3d) 极低出力段再校准
        ratio4 = out / cap
        tiny_zone = np.clip((0.10 - ratio4) / 0.10, 0.0, 1.0)
        shrink5 = 0.015 * tiny_zone * (0.3 + 0.7 * night_w)
        out = out * (1.0 - shrink5)

        # 3e) 日间中等出力段极轻微下压
        ratio5 = out / cap
        mid_zone = np.clip((ratio5 - 0.20) / 0.30, 0.0, 1.0) * np.clip((0.60 - ratio5) / 0.30, 0.0, 1.0)
        shrink6 = 0.008 * mid_zone * day_w
        out = out * (1.0 - shrink6)

        # 3f) 峰值削顶
        ratio6 = out / cap
        peak_zone = np.clip((ratio6 - 0.65) / 0.35, 0.0, 1.0)
        shrink7 = 0.010 * peak_zone * (0.5 + 0.5 * noon_w)
        out = out * (1.0 - shrink7)

        # 3g) 密集区间极轻微下压
        ratio7 = out / cap
        dense_zone = np.clip((ratio7 - 0.05) / 0.10, 0.0, 1.0) * np.clip((0.30 - ratio7) / 0.10, 0.0, 1.0)
        shrink8 = 0.005 * dense_zone
        out = out * (1.0 - shrink8)

        # 3h) 整体温和收缩
        ratio8 = out / cap
        global_zone = np.clip((ratio8 - 0.02) / 0.10, 0.0, 1.0)
        out = out * (1.0 - 0.006 * global_zone)

        # 3i) 针对 0.10~0.25 最密集区间进一步微调
        ratio9 = out / cap
        core_zone = np.clip((ratio9 - 0.10) / 0.05, 0.0, 1.0) * np.clip((0.25 - ratio9) / 0.05, 0.0, 1.0)
        out = out * (1.0 - 0.006 * core_zone)

        # 3j) 日间中等出力区间(0.15~0.45)进一步温和下压
        ratio10 = out / cap
        band_zone = np.clip((ratio10 - 0.12) / 0.06, 0.0, 1.0) * np.clip((0.45 - ratio10) / 0.08, 0.0, 1.0)
        shrink9 = 0.006 * band_zone * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink9)

        # 3k) 对 0.15~0.35 主密集带做更细的二次下压
        ratio11 = out / cap
        band2 = np.clip((ratio11 - 0.15) / 0.05, 0.0, 1.0) * np.clip((0.35 - ratio11) / 0.05, 0.0, 1.0)
        shrink10 = 0.004 * band2 * (0.4 + 0.6 * day_w)
        out = out * (1.0 - shrink10)

        # 3l) 傍晚过渡段(ratio 0.10~0.20)轻微下压
        ratio12 = out / cap
        dusk_zone = np.clip((ratio12 - 0.10) / 0.04, 0.0, 1.0) * np.clip((0.20 - ratio12) / 0.04, 0.0, 1.0)
        shrink11 = 0.004 * dusk_zone * dusk_w
        out = out * (1.0 - shrink11)

        # 3m) 中等偏上区间(0.35~0.55)极轻微下压
        ratio13 = out / cap
        upper_mid = np.clip((ratio13 - 0.35) / 0.05, 0.0, 1.0) * np.clip((0.55 - ratio13) / 0.05, 0.0, 1.0)
        shrink12 = 0.003 * upper_mid * (0.5 + 0.5 * noon_w)
        out = out * (1.0 - shrink12)

        # 3n) 极低出力带(0.02~0.08)温和下压
        ratio14 = out / cap
        near_zero = np.clip((ratio14 - 0.02) / 0.03, 0.0, 1.0) * np.clip((0.08 - ratio14) / 0.03, 0.0, 1.0)
        shrink13 = 0.010 * near_zero * (0.4 + 0.6 * night_w)
        out = out * (1.0 - shrink13)

        # 3o) 0.25~0.40 过渡带轻微下压
        ratio15 = out / cap
        trans_zone = np.clip((ratio15 - 0.25) / 0.05, 0.0, 1.0) * np.clip((0.40 - ratio15) / 0.05, 0.0, 1.0)
        shrink14 = 0.003 * trans_zone * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink14)

        # 3p) 0.08~0.15 次密集带轻微下压（夜间权重更高）
        ratio16 = out / cap
        low_band = np.clip((ratio16 - 0.08) / 0.03, 0.0, 1.0) * np.clip((0.15 - ratio16) / 0.03, 0.0, 1.0)
        shrink15 = 0.005 * low_band * (0.4 + 0.6 * night_w)
        out = out * (1.0 - shrink15)

        # 3q) 0.55~0.70 高段轻微削顶（正午权重更高）
        ratio17 = out / cap
        high_band = np.clip((ratio17 - 0.55) / 0.05, 0.0, 1.0) * np.clip((0.70 - ratio17) / 0.05, 0.0, 1.0)
        shrink16 = 0.004 * high_band * (0.5 + 0.5 * noon_w)
        out = out * (1.0 - shrink16)

        # 3r) 0.18~0.30 主密集核心带进一步温和下压（日间权重略高）
        ratio18 = out / cap
        core2 = np.clip((ratio18 - 0.18) / 0.04, 0.0, 1.0) * np.clip((0.30 - ratio18) / 0.04, 0.0, 1.0)
        shrink17 = 0.004 * core2 * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink17)

        # 3s) 0.30~0.45 中段补充轻微下压
        ratio19 = out / cap
        mid_band = np.clip((ratio19 - 0.30) / 0.04, 0.0, 1.0) * np.clip((0.45 - ratio19) / 0.04, 0.0, 1.0)
        shrink18 = 0.003 * mid_band * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink18)

        # 3t) 0.00~0.05 极近零带温和下压（夜间权重更高，避免虚高）
        ratio20 = out / cap
        zero_band = np.clip((0.05 - ratio20) / 0.05, 0.0, 1.0)
        shrink19 = 0.008 * zero_band * (0.3 + 0.7 * night_w)
        out = out * (1.0 - shrink19)

        # 3u) 正午高段(0.45~0.60)极轻微下压，抑制日间高估
        ratio21 = out / cap
        noon_high = np.clip((ratio21 - 0.45) / 0.05, 0.0, 1.0) * np.clip((0.60 - ratio21) / 0.05, 0.0, 1.0)
        shrink20 = 0.002 * noon_high * noon_w
        out = out * (1.0 - shrink20)

        # 3v) 0.12~0.18 密集过渡带轻微下压
        ratio22 = out / cap
        trans2 = np.clip((ratio22 - 0.12) / 0.03, 0.0, 1.0) * np.clip((0.18 - ratio22) / 0.03, 0.0, 1.0)
        shrink21 = 0.003 * trans2 * (0.4 + 0.6 * dusk_w)
        out = out * (1.0 - shrink21)

        # 3w) 整体再轻微收缩，主密集带(0.05~0.40)加权
        ratio23 = out / cap
        main_band = np.clip((ratio23 - 0.05) / 0.05, 0.0, 1.0) * np.clip((0.40 - ratio23) / 0.05, 0.0, 1.0)
        shrink22 = 0.004 * main_band
        out = out * (1.0 - shrink22)

        # 3x) 夜间极低段(0.00~0.04)额外温和下压
        ratio24 = out / cap
        night_tiny = np.clip((0.04 - ratio24) / 0.04, 0.0, 1.0)
        shrink23 = 0.006 * night_tiny * night_w
        out = out * (1.0 - shrink23)

        # 3y) 0.20~0.28 最核心密集带细粒度再下压
        ratio25 = out / cap
        core3 = np.clip((ratio25 - 0.20) / 0.03, 0.0, 1.0) * np.clip((0.28 - ratio25) / 0.03, 0.0, 1.0)
        shrink24 = 0.003 * core3 * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink24)

        # 3z) 新增：0.06~0.12 次低密集带温和下压（夜间权重更高）
        ratio26 = out / cap
        sub_low = np.clip((ratio26 - 0.06) / 0.02, 0.0, 1.0) * np.clip((0.12 - ratio26) / 0.02, 0.0, 1.0)
        shrink25 = 0.004 * sub_low * (0.4 + 0.6 * night_w)
        out = out * (1.0 - shrink25)

        # 3aa) 新增：0.22~0.32 核心带再次细粒度对齐
        ratio27 = out / cap
        core4 = np.clip((ratio27 - 0.22) / 0.03, 0.0, 1.0) * np.clip((0.32 - ratio27) / 0.03, 0.0, 1.0)
        shrink26 = 0.0025 * core4 * (0.5 + 0.5 * day_w)
        out = out * (1.0 - shrink26)

        # 3ab) 新增：整体主密集带(0.08~0.35)再微收
        ratio28 = out / cap
        main2 = np.clip((ratio28 - 0.08) / 0.04, 0.0, 1.0) * np.clip((0.35 - ratio28) / 0.04, 0.0, 1.0)
        shrink27 = 0.0025 * main2
        out = out * (1.0 - shrink27)

    # 4) 边界裁剪
    if not np.isnan(cap) and cap > 0:
        out = np.clip(out, 0.0, cap)
    else:
        out = np.clip(out, 0.0, None)

    # 5) 保底
    if np.nanstd(out) < 1e-9:
        out = pp.copy()

    return out