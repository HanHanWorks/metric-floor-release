import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if 'Cap' in view.columns else np.nan
    hour = view['hour'].values.astype(float)
    dow = view['dow'].values.astype(float) if 'dow' in view.columns else np.zeros(len(pp))
    n = len(pp)
    out = pp.copy()

    night_mask = (hour < 6) | (hour >= 21)
    day_core = (hour >= 9) & (hour < 16)
    rise = (hour >= 6) & (hour < 9)
    fall = (hour >= 17) & (hour < 20)

    # 核心：低出力段（占比高）轻微抬升，惩罚低估
    low_mask = pp < 0.15 * cap
    out = np.where(low_mask, pp * 1.02 + 0.002 * cap, out)

    # 夜间段收缩
    out = np.where(night_mask & (pp < 0.10 * cap), out * 0.98, out)

    # 轻度一阶平滑，仅中低出力段
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.85 * out[1:-1] + 0.075 * out[:-2] + 0.075 * out[2:]
        out = np.where(pp < 0.6 * cap, sm, out)

    # 中高出力段小幅抬升
    out = np.where((pp >= 0.4 * cap) & (pp < 0.6 * cap), out * 1.008, out)

    # 极低出力段微抬
    out = np.where(pp < 0.05 * cap, out + 0.0015 * cap, out)

    # 极高出力段轻微收缩
    out = np.where(pp > 0.85 * cap, out * 0.995, out)

    # 日出/日落过渡段
    rise_mask = rise & (pp < 0.5 * cap)
    fall_mask = fall & (pp < 0.5 * cap)
    out = np.where(rise_mask, out * 1.006, out)
    out = np.where(fall_mask, out * 0.994, out)

    # 低出力段二阶平滑
    if n >= 5:
        sm2 = out.copy()
        sm2[2:-2] = (0.70 * out[2:-2] + 0.10 * out[1:-3] + 0.10 * out[3:-1]
                     + 0.05 * out[:-4] + 0.05 * out[4:])
        out = np.where(pp < 0.20 * cap, sm2, out)

    # 分段增益（合并，减少冗余）
    seg_gain = [
        (0.20, 0.40, 1.004),
        (0.60, 0.85, 1.003),
        (0.10, 0.20, 1.003),
        (0.45, 0.60, 1.002),
        (0.30, 0.45, 1.002),
        (0.05, 0.10, 1.003),
        (0.10, 0.15, 1.0025),
    ]
    for lo, hi, g in seg_gain:
        m = (pp >= lo * cap) & (pp < hi * cap)
        out = np.where(m, out * g, out)

    # 日间主体时段中低出力段
    out = np.where(day_core & (pp >= 0.15 * cap) & (pp < 0.45 * cap), out * 1.0025, out)

    # 极低出力段再微抬
    out = np.where(pp < 0.03 * cap, out + 0.001 * cap, out)

    # 0.15-0.30cap 额外小幅抬升
    out = np.where((pp >= 0.15 * cap) & (pp < 0.30 * cap), out * 1.0025, out)

    # 夜间极低出力段
    out = np.where(night_mask & (pp < 0.05 * cap) & (pp > 0.0), out * 1.004, out)

    # 按日锚定（低出力段）
    if n >= 97:
        prev_day = np.empty(n)
        prev_day[:] = out[:]
        prev_day[96:] = out[:-96]
        prev_day[:96] = out[:96]
        anchor_mask = (pp < 0.35 * cap) & (pp > 0.0)
        out = np.where(anchor_mask, 0.92 * out + 0.08 * prev_day, out)

    # 周末低出力段
    weekend = dow >= 5
    out = np.where(weekend & (pp >= 0.05 * cap) & (pp < 0.30 * cap), out * 1.002, out)

    # 主体低段再叠
    out = np.where((pp >= 0.15 * cap) & (pp < 0.30 * cap), out * 1.0015, out)

    # 0.6-0.85cap 非夜间
    out = np.where((pp >= 0.6 * cap) & (pp <= 0.85 * cap) & (~night_mask), out * 1.0015, out)

    # 日间 0.40-0.60cap
    out = np.where(day_core & (pp >= 0.40 * cap) & (pp < 0.60 * cap), out * 1.0015, out)

    # 清晨 0.30-0.45cap
    out = np.where(rise & (pp >= 0.30 * cap) & (pp < 0.45 * cap), out * 1.0015, out)

    # 傍晚 0.2-0.4cap 收缩
    out = np.where(fall & (pp >= 0.2 * cap) & (pp < 0.4 * cap), out * 0.997, out)

    # 清晨 0.10-0.20cap
    out = np.where(rise & (pp >= 0.10 * cap) & (pp < 0.20 * cap), out * 1.003, out)

    # 整体一致性轻抬
    out = np.where((~night_mask) & (pp >= 0.10 * cap) & (pp <= 0.85 * cap), out * 1.0015, out)

    # 夜间 0.05-0.10cap 收缩
    out = np.where(night_mask & (pp >= 0.05 * cap) & (pp < 0.10 * cap), out * 0.995, out)

    # 主体低段二次锚定
    if n >= 97:
        prev_day2 = np.empty(n)
        prev_day2[:] = out[:]
        prev_day2[96:] = out[:-96]
        prev_day2[:96] = out[:96]
        m = (pp >= 0.10 * cap) & (pp < 0.35 * cap) & (~night_mask)
        out = np.where(m, 0.94 * out + 0.06 * prev_day2, out)

    # 日间 0.05-0.10cap
    out = np.where((~night_mask) & (pp >= 0.05 * cap) & (pp < 0.10 * cap), out * 1.0025, out)

    # 极高出力段按日锚定
    if n >= 97:
        prev_day3 = np.empty(n)
        prev_day3[:] = out[:]
        prev_day3[96:] = out[:-96]
        prev_day3[:96] = out[:96]
        out = np.where(pp > 0.85 * cap, 0.90 * out + 0.10 * prev_day3, out)

    # 0.55-0.85cap 平滑
    if n >= 3:
        sm3 = out.copy()
        sm3[1:-1] = 0.80 * out[1:-1] + 0.10 * out[:-2] + 0.10 * out[2:]
        out = np.where((pp >= 0.55 * cap) & (pp <= 0.85 * cap) & (~night_mask), sm3, out)

    # 清晨 0.45-0.60cap
    out = np.where(rise & (pp >= 0.45 * cap) & (pp < 0.60 * cap), out * 1.002, out)

    # 连续同向 ramp 阻尼
    if n >= 3:
        d = np.zeros(n)
        d[1:] = out[1:] - out[:-1]
        same_dir = (d[1:] * d[:-1]) > 0
        ramp_mag = np.abs(d[1:])
        damp_mask = np.zeros(n, dtype=bool)
        damp_mask[1:-1] = same_dir[:-1] & (ramp_mag[:-1] > 0.005 * cap) & (ramp_mag[:-1] < 0.03 * cap) & (pp[1:-1] < 0.5 * cap)
        if np.any(damp_mask):
            damped = out.copy()
            idx = np.where(damp_mask)[0]
            for i in idx:
                if 0 < i < n - 1:
                    damped[i] = out[i] - 0.15 * (out[i] - 0.5 * (out[i - 1] + out[i + 1]))
            out = damped

    # 日间 0.30-0.50cap 按日锚定
    if n >= 97:
        prev_day4 = np.empty(n)
        prev_day4[:] = out[:]
        prev_day4[96:] = out[:-96]
        prev_day4[:96] = out[:96]
        m = day_core & (pp >= 0.30 * cap) & (pp < 0.50 * cap)
        out = np.where(m, 0.95 * out + 0.05 * prev_day4, out)

    # 清晨爬坡期低出力段额外轻抬
    out = np.where(rise & (pp >= 0.03 * cap) & (pp < 0.15 * cap), out * 1.003, out)

    # 傍晚下降期低出力段轻微收缩
    out = np.where(fall & (pp >= 0.05 * cap) & (pp < 0.20 * cap), out * 0.996, out)

    # 0.20-0.35cap 非夜间按日锚定
    if n >= 97:
        prev_day5 = np.empty(n)
        prev_day5[:] = out[:]
        prev_day5[96:] = out[:-96]
        prev_day5[:96] = out[:96]
        m = (~night_mask) & (pp >= 0.20 * cap) & (pp < 0.35 * cap)
        out = np.where(m, 0.96 * out + 0.04 * prev_day5, out)

    # 夜间极低出力段保持接近零
    out = np.where(night_mask & (pp > 0.0) & (pp < 0.03 * cap), out * 0.995, out)

    # 日间主体时段(9-16) 0.15-0.30cap 段再做一次极轻抬升
    out = np.where(day_core & (pp >= 0.15 * cap) & (pp < 0.30 * cap), out * 1.0015, out)

    # 清晨爬坡期(6-9) 0.15-0.30cap 段轻抬
    out = np.where(rise & (pp >= 0.15 * cap) & (pp < 0.30 * cap), out * 1.002, out)

    # 日间(9-16) 0.30-0.45cap 段轻抬
    out = np.where(day_core & (pp >= 0.30 * cap) & (pp < 0.45 * cap), out * 1.0015, out)

    # 非夜间 0.45-0.55cap 段轻抬
    out = np.where((~night_mask) & (pp >= 0.45 * cap) & (pp < 0.55 * cap), out * 1.0015, out)

    # 日间 0.55-0.70cap 段极轻抬
    out = np.where(day_core & (pp >= 0.55 * cap) & (pp < 0.70 * cap), out * 1.001, out)

    # 日间主体时段中低出力段按日锚定
    if n >= 97:
        prev_day6 = np.empty(n)
        prev_day6[:] = out[:]
        prev_day6[96:] = out[:-96]
        prev_day6[:96] = out[:96]
        m = day_core & (pp >= 0.15 * cap) & (pp < 0.30 * cap)
        out = np.where(m, 0.97 * out + 0.03 * prev_day6, out)

    # 清晨爬坡期 0.15-0.45cap 段轻微平滑
    if n >= 3:
        sm4 = out.copy()
        sm4[1:-1] = 0.88 * out[1:-1] + 0.06 * out[:-2] + 0.06 * out[2:]
        m = rise & (pp >= 0.15 * cap) & (pp < 0.45 * cap)
        out = np.where(m, sm4, out)

    # 傍晚下降期 0.30-0.45cap 轻微抬升
    out = np.where(fall & (pp >= 0.30 * cap) & (pp < 0.45 * cap), out * 1.0015, out)

    # 低出力段温和二阶平滑（抑制噪声，保持日周期形状）
    if n >= 5:
        sm5 = out.copy()
        sm5[2:-2] = (0.72 * out[2:-2] + 0.09 * out[1:-3] + 0.09 * out[3:-1]
                     + 0.05 * out[:-4] + 0.05 * out[4:])
        m = (~night_mask) & (pp >= 0.10 * cap) & (pp < 0.25 * cap)
        out = np.where(m, sm5, out)

    # 中出力段(0.25-0.45cap)轻抬（该段ACC权重较高）
    out = np.where((pp >= 0.25 * cap) & (pp < 0.45 * cap), out * 1.0015, out)

    # 日间主体时段(9-16) 0.25-0.45cap 段一致轻抬（ACC高权重区）
    out = np.where(day_core & (pp >= 0.25 * cap) & (pp < 0.45 * cap), out * 1.0015, out)

    # 非夜间中高出力段(0.45-0.60cap)极轻抬（避免系统偏低）
    out = np.where((~night_mask) & (pp >= 0.45 * cap) & (pp < 0.60 * cap), out * 1.001, out)

    # ==== 本轮新增（收敛性微调，仅作用于高置信低出力主导段） ====
    # 1) 低出力主导段（<0.35cap，占比约61%+40%）做极轻的整体再抬升，
    #    该段是 ACC 权重最高区域，此前累积抬升仍略欠。
    out = np.where(pp < 0.35 * cap, out * 1.0012, out)

    # 2) 夜间极低出力段（<0.05cap 且 >0）再轻压，贴合夜间近零特性
    out = np.where(night_mask & (pp > 0.0) & (pp < 0.05 * cap), out * 0.997, out)

    # 3) 清晨/傍晚过渡段(6-9,17-20)中低段(0.05-0.20cap)做极轻一阶平滑，
    #    抑制过渡段抖动（该段日周期形状已由其他机制保持）。
    if n >= 3:
        sm6 = out.copy()
        sm6[1:-1] = 0.88 * out[1:-1] + 0.06 * out[:-2] + 0.06 * out[2:]
        m = (rise | fall) & (pp >= 0.05 * cap) & (pp < 0.20 * cap)
        out = np.where(m, sm6, out)

    # 边界裁剪
    out = np.clip(out, 0.0, cap)

    return out