import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 1) 轻度平滑，抑制 15 分钟级噪声
    if n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    r = np.clip(out / cap, 0.0, 1.0)

    # 2) 分段收缩：低段过预测惩罚最重；本轮在前几轮基础上
    #    继续小幅加深低段收缩，中高段保持略微收紧但避免过度压缩。
    #    第39轮：低段与中低段进一步轻微加深，极低段略微回调避免过度压低。
    very_low = r < 0.05
    r[very_low] = r[very_low] * 0.712

    low1 = (r >= 0.05) & (r < 0.12)
    r[low1] = r[low1] * 0.842

    low2 = (r >= 0.12) & (r < 0.20)
    r[low2] = r[low2] * 0.884

    mid = (r >= 0.20) & (r < 0.35)
    r[mid] = r[mid] * 0.948

    mid_high = (r >= 0.35) & (r < 0.50)
    r[mid_high] = r[mid_high] * 0.920

    high = (r >= 0.50) & (r < 0.60)
    r[high] = r[high] * 0.894

    very_high = r >= 0.60
    r[very_high] = r[very_high] * 0.868

    out = r * cap

    # 3) 物理约束
    out = np.clip(out, 0.0, cap)

    # 4) 防止常数序列
    if n > 1 and np.allclose(out, out[0]):
        out = out + 1e-6 * np.arange(n)

    return out