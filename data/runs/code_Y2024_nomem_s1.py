import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    n = len(pp)
    out = pp.copy()

    # 1) 轻微时间平滑，抑制 15 分钟噪声但保留日周期形态
    #    权重更保守（中心权重更高），避免过度平滑掉真实的爬坡
    if n >= 7:
        sm = pp.copy()
        sm[3:-3] = (0.05 * pp[:-6] + 0.10 * pp[1:-5] + 0.20 * pp[2:-4]
                    + 0.30 * pp[3:-3]
                    + 0.20 * pp[4:-2] + 0.10 * pp[5:-1] + 0.05 * pp[6:])
        # 边缘用较轻的平滑
        sm[1] = 0.7 * pp[1] + 0.3 * pp[0]
        sm[2] = 0.6 * pp[2] + 0.25 * pp[1] + 0.15 * pp[3]
        sm[-2] = 0.7 * pp[-2] + 0.3 * pp[-1]
        sm[-3] = 0.6 * pp[-3] + 0.25 * pp[-2] + 0.15 * pp[-4]
        out = sm
    elif n >= 5:
        sm = pp.copy()
        sm[2:-2] = (0.15 * pp[:-4] + 0.2 * pp[1:-3] + 0.3 * pp[2:-2]
                    + 0.2 * pp[3:-1] + 0.15 * pp[4:])
        out = sm
    elif n >= 3:
        sm = pp.copy()
        sm[1:-1] = 0.25 * pp[:-2] + 0.5 * pp[1:-1] + 0.25 * pp[2:]
        out = sm

    # 2) 低出力段：极低值(<0.02cap)轻微压缩，减少小值抖动带来的绝对误差
    #    阈值更低、幅度更小，避免系统性低估
    low_mask = out < 0.02 * cap
    out[low_mask] = out[low_mask] * 0.95

    # 3) 非负裁剪
    out = np.clip(out, 0.0, cap)

    # 4) 保证不是常数序列
    if n > 1 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out