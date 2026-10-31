import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    x = pp / cap
    n = len(x)
    if n == 0:
        return pp

    # 归一化序列处理
    xs = x.copy()

    # 1) 极轻平滑：仅对内部点做 3 点加权，权重偏向中心，几乎不改变爬坡
    if n >= 3:
        sm = xs.copy()
        sm[1:-1] = 0.15 * xs[:-2] + 0.70 * xs[1:-1] + 0.15 * xs[2:]
        xs = sm

    # 2) 低功率段收缩：电网 ACC 对低功率误差敏感，PP 有 51% 低于 0.1cap
    #    对极低段略收缩，保留 0 附近结构
    low1 = xs < 0.02
    xs = np.where(low1, xs * 0.80, xs)
    low2 = (xs >= 0.02) & (xs < 0.06)
    xs = np.where(low2, xs * 0.90, xs)

    # 3) 高值软压缩：p99≈0.425，无 >0.9cap，压缩超过 0.40 的尾部
    hi = xs > 0.40
    xs = np.where(hi, 0.40 + (xs - 0.40) * 0.85, xs)

    # 4) 日内平滑：用 5 点核仅在内部区域做很轻的平滑，混合比例小以保留爬坡
    if n >= 9:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
        conv = np.convolve(xs, kernel, mode='same')
        conv[:2] = xs[:2]
        conv[-2:] = xs[-2:]
        # 混合：85% 原序列 + 15% 平滑，避免抹平真实细节
        xs = 0.85 * xs + 0.15 * conv

    # 5) 单调性/边界保护
    xs = np.clip(xs, 0.0, 1.0)

    out = xs * cap

    # 防止常数输出
    if n > 1 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out