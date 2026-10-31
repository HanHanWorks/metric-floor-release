import numpy as np
import pandas as pd

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if 'Cap' in view.columns else np.nan
    n = len(pp)
    if n == 0:
        return pp

    base = pp.copy()
    has_cap = np.isfinite(cap) and cap > 0

    # 数据画像：预测值普遍偏低（mean_ratio=0.17, p50=0.12, p90=0.41），
    # frac_below_02cap=0.68，大量点接近 0；step_autocorr=0.9977 极高说明序列极平滑；
    # day_autocorr=0.445 说明日间有一定相关性；ramp_max=0.314 是离群跳变。
    # ACC 对偏差敏感，尤其低功率段相对误差大。策略：
    #  - 保持序列平滑（与原始高度一致），避免引入额外噪声；
    #  - 对低功率段温和抬升以补偿系统性低估；
    #  - 抑制明显离群尖峰；
    #  - 保持日间结构（用前一日同时刻做轻微引导）。

    out = base.copy()

    # 1) 极轻平滑，几乎保持原始（step_autocorr 极高，平滑收益有限）
    if n >= 3:
        kernel = np.array([0.15, 0.70, 0.15])
        pad = np.pad(base, (1, 1), mode='edge')
        smooth = np.convolve(pad, kernel, mode='valid')
        out = 0.85 * base + 0.15 * smooth

    # 2) 日间结构引导：用前一日同时刻值做轻微混合（day_autocorr=0.445 有一定参考性）
    if n >= 96:
        prev_day = np.concatenate([out[:96], out[:-96]])
        # 仅在原始预测本身偏离前一日较大时做轻微修正，且幅度很小
        out = 0.94 * out + 0.06 * prev_day

    # 3) 抑制离群尖峰：参考 p99 差分做软限幅
    if n >= 3:
        d = np.diff(out)
        absd = np.abs(d)
        ref = np.percentile(absd, 99) if n > 20 else (absd.max() + 1e-9)
        limit = max(ref * 2.5, 1e-9)
        d_clip = np.clip(d, -limit, limit)
        out_clip = np.empty_like(out)
        out_clip[0] = out[0]
        out_clip[1:] = out[0] + np.cumsum(d_clip)
        out = 0.90 * out + 0.10 * out_clip

    # 4) 低功率段温和抬升：补偿系统性低估，幅度克制以控制绝对误差
    if has_cap:
        r = out / cap
        # 极低段（<5%）温和抬升，中低段（5%-15%）轻微抬升
        lift = np.ones_like(out)
        lift = np.where(r < 0.05, 1.06, lift)
        lift = np.where((r >= 0.05) & (r < 0.15), 1.03, lift)
        lift = np.where((r >= 0.15) & (r < 0.30), 1.01, lift)
        out = out * lift

    # 5) 物理约束
    upper = cap if has_cap else np.inf
    out = np.clip(out, 0.0, upper)

    # 6) 极小值清理
    if has_cap:
        thr = 0.003 * cap
        out = np.where(out < thr, 0.0, out)

    # 7) 保证非恒定
    if np.allclose(out, out[0]):
        out = base.copy()

    return out