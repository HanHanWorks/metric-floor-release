import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].values[0]) if len(view) else 80.0
    if cap <= 0:
        cap = 80.0
    n = len(pp)
    if n == 0:
        return pp.copy()

    out = pp.copy()

    # 1) 低出力收缩：分段收缩（保留已验证结构，比例微调更贴近低段偏高修正）
    ratios = np.array([0.50, 0.60, 0.71, 0.83, 0.91, 0.96, 0.985])
    bounds = np.array([0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.70]) * cap
    prev = 0.0
    for b, r in zip(bounds, ratios):
        m = (out >= prev) & (out < b)
        out[m] = out[m] * r
        prev = b

    # 2) 轻度平滑：5点加权滑动平均（边界用对称复制避免边缘跳变）
    if n >= 5:
        ext = np.concatenate(([out[0], out[0]], out, [out[-1], out[-1]]))
        sm = out.copy()
        sm[:] = (0.12 * ext[:-4] + 0.20 * ext[1:-3] + 0.36 * ext[2:-2]
                 + 0.20 * ext[3:-1] + 0.12 * ext[4:])
        out = sm
    elif n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 3) 削峰：极高段下压（更温和，保留峰值形态）
    high_mask = out > 0.82 * cap
    out[high_mask] = 0.82 * cap + (out[high_mask] - 0.82 * cap) * 0.66

    # 4) 爬坡限制（分时段自适应，避免过度压平白天平滑输出）
    if n >= 2:
        hour = view['hour'].values.astype(float) if 'hour' in view.columns else np.full(n, 12.0)
        for i in range(1, n):
            h = hour[i]
            if 9.0 <= h <= 15.0:
                max_ramp = 0.12 * cap
            elif 7.0 <= h < 9.0 or 15.0 < h <= 18.0:
                max_ramp = 0.09 * cap
            else:
                max_ramp = 0.05 * cap
            diff = out[i] - out[i - 1]
            if diff > max_ramp:
                out[i] = out[i - 1] + max_ramp
            elif diff < -max_ramp:
                out[i] = out[i - 1] - max_ramp

    # 5) 非负裁剪与容量约束
    out = np.clip(out, 0.0, cap)

    # 6) 防止常数输出
    if n > 1 and np.allclose(out, out[0]):
        out = out + np.linspace(0, 1e-6, n)

    return out