import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else float(np.nanmax(pp))
    hour = view['hour'].values.astype(float)
    n = len(pp)

    out = pp.copy()

    # 1) 轻度 3 点平滑，仅作用于低出力段（保留爬坡）
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        low_mask = out < 0.25 * cap
        out = np.where(low_mask, sm, out)

    # 2) 夜间收缩（数据画像：夜间预测偏高，向 0 收缩）
    night = (hour < 6.0) | (hour >= 20.0)
    out = np.where(night, out * 0.85, out)

    # 3) 极低出力段进一步收缩
    tiny = out < 0.05 * cap
    out = np.where(tiny, out * 0.70, out)

    # 4) 白天中低出力段轻微收缩
    mid_low = (~night) & (out >= 0.05 * cap) & (out < 0.20 * cap)
    out = np.where(mid_low, out * 0.94, out)

    # 5) 中等出力段温和收缩
    mid = (out >= 0.20 * cap) & (out < 0.40 * cap)
    out = np.where(mid, out * 0.96, out)

    # 6) 过渡段再轻微收缩
    trans = (~night) & (((hour >= 6.0) & (hour < 8.0)) | ((hour >= 17.0) & (hour < 20.0)))
    out = np.where(trans, out * 0.95, out)

    # 7) 整体偏低分布：对 0.40cap 以下全局温和收缩
    below_mid = out < 0.40 * cap
    out = np.where(below_mid, out * 0.97, out)

    # 8) 极低出力段向零靠拢
    near_zero = out < 0.02 * cap
    out = np.where(near_zero, out * 0.80, out)

    # 9) 裁剪
    out = np.clip(out, 0.0, cap)

    # 10) 防常数
    if np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out