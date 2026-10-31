import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 100.0
    n = len(pp)
    
    out = pp.copy()
    
    # 1) 去毛刺：仅在偏离局部中值较大时收敛，保留真实爬坡
    if n >= 3:
        p = np.pad(pp, 1, mode='edge')
        med = np.array([np.median(p[i:i+3]) for i in range(n)])
        dev = pp - med
        thresh = 0.015 * cap
        adj = np.where(np.abs(dev) > thresh, med + np.sign(dev) * thresh, pp)
        out = 0.6 * out + 0.4 * adj
    
    # 2) 低出力段温和抬升：pp 分布严重偏低(63%<0.2cap, 36%<0.1cap)
    #    电网 ACC 口径下低出力段绝对误差权重相对小，但系统性低估会拉低准确率。
    #    对极低值做温和抬升，避免常数化，幅度比之前更保守以防过冲。
    low_mask = pp < 0.04 * cap
    lift = 0.008 * cap * (1.0 - pp / (0.04 * cap + 1e-6))
    out = np.where(low_mask, out + lift, out)
    
    # 3) 爬坡限幅：ramp_p99≈0.043cap, ramp_max≈0.276cap
    #    用稍宽的上限保留真实大爬坡，同时抑制不真实尖峰
    max_ramp = 0.07 * cap
    for i in range(1, n):
        d = out[i] - out[i-1]
        if d > max_ramp:
            out[i] = out[i-1] + max_ramp
        elif d < -max_ramp:
            out[i] = out[i-1] - max_ramp
    
    # 4) 物理约束
    out = np.clip(out, 0.0, cap)
    
    # 5) 防止常数序列
    if np.std(out) < 1e-6:
        out = out + 0.001 * cap * np.sin(np.arange(n) * 2 * np.pi / 96.0)
    
    return out