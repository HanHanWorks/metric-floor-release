import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    n = len(pp)
    out = pp.copy()

    if n < 3:
        return np.clip(out, 0.0, cap)

    # 1) 极轻平滑：序列高度平滑，仅去单点噪声
    w = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
    sm = out.copy()
    for i in range(2, n - 2):
        sm[i] = np.dot(w, out[i-2:i+3])
    out = sm

    # 2) 物理边界
    out = np.clip(out, 0.0, cap)

    # 3) 极低值轻微收缩向 0
    tiny = out < 0.01 * cap
    out[tiny] = out[tiny] * 0.5

    # 4) 孤立小尖峰抑制
    for i in range(1, n - 1):
        if out[i] < 0.02 * cap and out[i] > out[i-1] and out[i] > out[i+1]:
            out[i] = 0.5 * (out[i-1] + out[i+1])

    # 5) 接近满发点轻微下压
    hi = out > 0.85 * cap
    out[hi] = 0.85 * cap + 0.5 * (out[hi] - 0.85 * cap)

    # 6) 中高段(0.5~0.85cap)轻微向下收缩
    mid = (out > 0.5 * cap) & (out <= 0.85 * cap)
    out[mid] = out[mid] - 0.02 * (out[mid] - 0.5 * cap)

    # 7) 低中段(0.05~0.2cap)轻微下压
    lowmid = (out > 0.05 * cap) & (out <= 0.2 * cap)
    out[lowmid] = out[lowmid] - 0.01 * (out[lowmid] - 0.05 * cap)

    # 8) 中低段(0.2~0.5cap)极轻微下压
    lowmid2 = (out > 0.2 * cap) & (out <= 0.5 * cap)
    out[lowmid2] = out[lowmid2] - 0.008 * (out[lowmid2] - 0.2 * cap)

    # 9) 全局极轻向下偏移
    nz = out > 0.02 * cap
    out[nz] = out[nz] - 0.003 * cap

    # 10) 密集段(0.1~0.2cap)二次微调
    dense = (out > 0.1 * cap) & (out <= 0.2 * cap)
    out[dense] = out[dense] - 0.006 * (out[dense] - 0.1 * cap)

    # 11) 低段(0.02~0.05cap)温和收缩
    lowseg = (out > 0.02 * cap) & (out <= 0.05 * cap)
    out[lowseg] = out[lowseg] - 0.005 * (out[lowseg] - 0.02 * cap)

    # 12) 0.2~0.3cap 二次下压
    seg2 = (out > 0.2 * cap) & (out <= 0.3 * cap)
    out[seg2] = out[seg2] - 0.004 * (out[seg2] - 0.2 * cap)

    # 13) 0.3~0.5cap 二次下压
    seg3 = (out > 0.3 * cap) & (out <= 0.5 * cap)
    out[seg3] = out[seg3] - 0.003 * (out[seg3] - 0.3 * cap)

    # 14) 0.05~0.1cap 二次下压
    seg4 = (out > 0.05 * cap) & (out <= 0.1 * cap)
    out[seg4] = out[seg4] - 0.004 * (out[seg4] - 0.05 * cap)

    # 15) 0.5~0.7cap 极轻二次下压
    seg5 = (out > 0.5 * cap) & (out <= 0.7 * cap)
    out[seg5] = out[seg5] - 0.004 * (out[seg5] - 0.5 * cap)

    # 16) 0.7~0.85cap 极轻下压
    seg6 = (out > 0.7 * cap) & (out <= 0.85 * cap)
    out[seg6] = out[seg6] - 0.006 * (out[seg6] - 0.7 * cap)

    # 17) 0.4~0.5cap 极轻下压
    seg7 = (out > 0.4 * cap) & (out <= 0.5 * cap)
    out[seg7] = out[seg7] - 0.003 * (out[seg7] - 0.4 * cap)

    # 18) 0.15~0.25cap 密集带极轻下压
    seg8 = (out > 0.15 * cap) & (out <= 0.25 * cap)
    out[seg8] = out[seg8] - 0.003 * (out[seg8] - 0.15 * cap)

    # 19) 0.25~0.35cap 过渡带极轻下压
    seg9 = (out > 0.25 * cap) & (out <= 0.35 * cap)
    out[seg9] = out[seg9] - 0.0035 * (out[seg9] - 0.25 * cap)

    # 20) 0.08~0.12cap 核心密度带极轻下压
    seg10 = (out > 0.08 * cap) & (out <= 0.12 * cap)
    out[seg10] = out[seg10] - 0.004 * (out[seg10] - 0.08 * cap)

    # 21) 0.35~0.45cap 过渡带极轻下压
    seg11 = (out > 0.35 * cap) & (out <= 0.45 * cap)
    out[seg11] = out[seg11] - 0.0025 * (out[seg11] - 0.35 * cap)

    # 22) 0.6~0.7cap 中高带极轻二次下压
    seg12 = (out > 0.6 * cap) & (out <= 0.7 * cap)
    out[seg12] = out[seg12] - 0.003 * (out[seg12] - 0.6 * cap)

    # 23) 0.12~0.18cap 密度峰核心带极轻下压
    seg13 = (out > 0.12 * cap) & (out <= 0.18 * cap)
    out[seg13] = out[seg13] - 0.0035 * (out[seg13] - 0.12 * cap)

    # 24) 0.45~0.55cap 中段过渡带极轻下压
    seg14 = (out > 0.45 * cap) & (out <= 0.55 * cap)
    out[seg14] = out[seg14] - 0.002 * (out[seg14] - 0.45 * cap)

    # 25) 0.18~0.22cap 密度峰肩部带极轻下压
    seg15 = (out > 0.18 * cap) & (out <= 0.22 * cap)
    out[seg15] = out[seg15] - 0.003 * (out[seg15] - 0.18 * cap)

    # 26) 0.55~0.65cap 中高衔接带极轻下压
    seg16 = (out > 0.55 * cap) & (out <= 0.65 * cap)
    out[seg16] = out[seg16] - 0.0025 * (out[seg16] - 0.55 * cap)

    # 27) 0.03~0.06cap 低值起始带极轻下压
    seg17 = (out > 0.03 * cap) & (out <= 0.06 * cap)
    out[seg17] = out[seg17] - 0.003 * (out[seg17] - 0.03 * cap)

    # 28) 0.06~0.09cap 低值过渡带极轻下压
    seg18 = (out > 0.06 * cap) & (out <= 0.09 * cap)
    out[seg18] = out[seg18] - 0.0035 * (out[seg18] - 0.06 * cap)

    # 29) 0.22~0.28cap 密度峰后过渡带极轻下压
    seg19 = (out > 0.22 * cap) & (out <= 0.28 * cap)
    out[seg19] = out[seg19] - 0.003 * (out[seg19] - 0.22 * cap)

    # 30) 0.65~0.75cap 中高带极轻下压
    seg20 = (out > 0.65 * cap) & (out <= 0.75 * cap)
    out[seg20] = out[seg20] - 0.0025 * (out[seg20] - 0.65 * cap)

    # 31) 0.09~0.11cap 核心密度峰极轻下压
    seg21 = (out > 0.09 * cap) & (out <= 0.11 * cap)
    out[seg21] = out[seg21] - 0.003 * (out[seg21] - 0.09 * cap)

    # 32) 0.28~0.32cap 峰后肩部带极轻下压
    seg22 = (out > 0.28 * cap) & (out <= 0.32 * cap)
    out[seg22] = out[seg22] - 0.0025 * (out[seg22] - 0.28 * cap)

    # 33) 0.75~0.82cap 高段极轻下压
    seg23 = (out > 0.75 * cap) & (out <= 0.82 * cap)
    out[seg23] = out[seg23] - 0.002 * (out[seg23] - 0.75 * cap)

    # 34) 0.11~0.15cap 密度峰右肩带极轻下压
    seg24 = (out > 0.11 * cap) & (out <= 0.15 * cap)
    out[seg24] = out[seg24] - 0.0025 * (out[seg24] - 0.11 * cap)

    # 35) 主工作区整体极轻下移
    main = (out > 0.2 * cap) & (out <= 0.6 * cap)
    out[main] = out[main] - 0.0015 * cap

    # 36) 对 0.02~0.2cap 低值主密度区做一次温和整体下移
    lowmain = (out > 0.02 * cap) & (out <= 0.2 * cap)
    out[lowmain] = out[lowmain] - 0.002 * cap

    # 37) 对 0.2~0.35cap 次密度带再做一次极轻下移
    dens2 = (out > 0.2 * cap) & (out <= 0.35 * cap)
    out[dens2] = out[dens2] - 0.001 * cap

    # 38) 对核心低值密度带(0.08~0.20cap)再做一次极轻下移
    corelow = (out > 0.08 * cap) & (out <= 0.20 * cap)
    out[corelow] = out[corelow] - 0.0008 * cap

    # 39) 对 0.20~0.30cap 次核心带做一次极轻下移
    coremid = (out > 0.20 * cap) & (out <= 0.30 * cap)
    out[coremid] = out[coremid] - 0.0006 * cap

    # 40) 对 0.10~0.18cap 最高密度核心带再做一次极轻下移
    peak = (out > 0.10 * cap) & (out <= 0.18 * cap)
    out[peak] = out[peak] - 0.0005 * cap

    # 41) 对 0.18~0.25cap 密度峰肩部带再做一次极轻下移
    shoulder = (out > 0.18 * cap) & (out <= 0.25 * cap)
    out[shoulder] = out[shoulder] - 0.0004 * cap

    # 42) 对 0.12~0.16cap 密度峰最核心带做极轻下移
    core = (out > 0.12 * cap) & (out <= 0.16 * cap)
    out[core] = out[core] - 0.0004 * cap

    # 43) 对 0.16~0.20cap 密度峰上肩带做极轻下移
    upper = (out > 0.16 * cap) & (out <= 0.20 * cap)
    out[upper] = out[upper] - 0.0003 * cap

    # 44) 对 0.10~0.14cap 密度峰下肩带做极轻下移
    lower = (out > 0.10 * cap) & (out <= 0.14 * cap)
    out[lower] = out[lower] - 0.0003 * cap

    # 45) 对 0.14~0.18cap 密度峰中带做极轻下移
    midpeak = (out > 0.14 * cap) & (out <= 0.18 * cap)
    out[midpeak] = out[midpeak] - 0.00025 * cap

    # 46) 对 0.04~0.08cap 低值密集带做极轻下移
    lowdense = (out > 0.04 * cap) & (out <= 0.08 * cap)
    out[lowdense] = out[lowdense] - 0.0004 * cap

    # 47) 对 0.08~0.10cap 密度峰左肩带做极轻下移
    lshoulder = (out > 0.08 * cap) & (out <= 0.10 * cap)
    out[lshoulder] = out[lshoulder] - 0.0004 * cap

    # 48) 对 0.30~0.40cap 中段主工作带做极轻下移
    midmain = (out > 0.30 * cap) & (out <= 0.40 * cap)
    out[midmain] = out[midmain] - 0.0004 * cap

    # 49) 新增：对 0.02~0.04cap 极低值带做轻微收缩（该带预测易偏高）
    verylow = (out > 0.02 * cap) & (out <= 0.04 * cap)
    out[verylow] = out[verylow] * 0.985

    # 50) 新增：对 0.40~0.50cap 中高过渡带做极轻下移
    midhigh = (out > 0.40 * cap) & (out <= 0.50 * cap)
    out[midhigh] = out[midhigh] - 0.0003 * cap

    # 51) 新增：对 0.50~0.60cap 中高带做极轻下移
    highmid = (out > 0.50 * cap) & (out <= 0.60 * cap)
    out[highmid] = out[highmid] - 0.0003 * cap

    # 52) 最终裁剪
    out = np.clip(out, 0.0, cap)

    # 53) 退化保护
    if np.allclose(out, out[0]):
        out = np.clip(pp.copy(), 0.0, cap)

    return out