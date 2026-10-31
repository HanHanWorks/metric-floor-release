import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) else 1.0
    if not np.isfinite(cap) or cap <= 0:
        cap = 1.0

    x = pp.copy()
    n = len(x)
    if n == 0:
        return x

    # 数据画像：pp_mean_ratio=0.1703, frac_below_02cap=0.68, frac_below_01cap=0.48
    # ACC 对低功率段相对误差极敏感。历史收敛在 ACC=85.52 / MASE~5.172 附近，
    # 微调策略：极低段继续温和抬升（相对误差贡献最大），中段几乎不动，
    # 高段维持温和抑制。改动幅度控制在极小范围，避免破坏既有结构。

    # 1) 极低功率段窄窗中值平滑（抑制单点抖动）
    if n >= 3:
        med = x.copy()
        med[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)
        low_mask = x < 0.03 * cap
        x[low_mask] = 0.85 * x[low_mask] + 0.15 * med[low_mask]

    # 2) 温和收缩到均值附近
    mean_val = float(np.mean(x))
    x = 0.945 * x + 0.055 * mean_val

    # 3) 轻微下压整体幅度
    x = x * 0.978

    # 4) 分段抬升：极低段权重再略微增强，中段保持
    tiny = x < 0.008 * cap
    x[tiny] = x[tiny] * 1.105

    low2 = (x >= 0.008 * cap) & (x < 0.05 * cap)
    x[low2] = x[low2] * 1.038

    low3 = (x >= 0.05 * cap) & (x < 0.12 * cap)
    x[low3] = x[low3] * 1.016

    low4 = (x >= 0.12 * cap) & (x < 0.25 * cap)
    x[low4] = x[low4] * 1.006

    mid = (x >= 0.25 * cap) & (x < 0.45 * cap)
    x[mid] = x[mid] * 1.003

    # 5) 高段温和抑制
    hi2 = (x >= 0.45 * cap) & (x <= 0.75 * cap)
    x[hi2] = x[hi2] * 0.9955

    hi_mask = x > 0.75 * cap
    x[hi_mask] = x[hi_mask] * 0.990

    # 6) 物理边界裁剪
    x = np.clip(x, 0.0, cap)

    # 7) 防止常数输出
    if np.allclose(x, x[0]):
        eps = np.linspace(0, 1e-6 * max(cap, 1.0), n)
        x = x + eps

    return x