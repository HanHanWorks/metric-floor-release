import numpy as np

def transform(view):
    pp = view['PP'].values.astype(float)
    cap = float(view['Cap'].iloc[0]) if len(view) > 0 else 1.0
    if cap <= 0:
        cap = 1.0

    n = len(pp)
    if n == 0:
        return pp

    out = pp.copy()

    # 平滑：13 点二项式平滑（完整边界处理）
    if n >= 13:
        sm = out.copy()
        sm[6:-6] = (1.0 * out[:-12] + 12.0 * out[1:-11] + 66.0 * out[2:-10]
                    + 220.0 * out[3:-9] + 495.0 * out[4:-8] + 792.0 * out[5:-7]
                    + 924.0 * out[6:-6]
                    + 792.0 * out[7:-5] + 495.0 * out[8:-4] + 220.0 * out[9:-3]
                    + 66.0 * out[10:-2] + 12.0 * out[11:-1] + 1.0 * out[12:]) / 4096.0
        sm[1] = 0.25 * out[0] + 0.5 * out[1] + 0.25 * out[2]
        sm[-2] = 0.25 * out[-3] + 0.5 * out[-2] + 0.25 * out[-1]
        sm[2] = (1.0 * out[0] + 4.0 * out[1] + 6.0 * out[2]
                 + 4.0 * out[3] + 1.0 * out[4]) / 16.0
        sm[-3] = (1.0 * out[-5] + 4.0 * out[-4] + 6.0 * out[-3]
                  + 4.0 * out[-2] + 1.0 * out[-1]) / 16.0
        sm[3] = (1.0 * out[0] + 6.0 * out[1] + 15.0 * out[2]
                 + 20.0 * out[3] + 15.0 * out[4] + 6.0 * out[5]
                 + 1.0 * out[6]) / 64.0
        sm[-4] = (1.0 * out[-7] + 6.0 * out[-6] + 15.0 * out[-5]
                  + 20.0 * out[-4] + 15.0 * out[-3] + 6.0 * out[-2]
                  + 1.0 * out[-1]) / 64.0
        sm[4] = (1.0 * out[0] + 8.0 * out[1] + 28.0 * out[2]
                 + 56.0 * out[3] + 70.0 * out[4] + 56.0 * out[5]
                 + 28.0 * out[6] + 8.0 * out[7] + 1.0 * out[8]) / 256.0
        sm[-5] = (1.0 * out[-9] + 8.0 * out[-8] + 28.0 * out[-7]
                  + 56.0 * out[-6] + 70.0 * out[-5] + 56.0 * out[-4]
                  + 28.0 * out[-3] + 8.0 * out[-2] + 1.0 * out[-1]) / 256.0
        sm[5] = (1.0 * out[0] + 10.0 * out[1] + 45.0 * out[2]
                 + 120.0 * out[3] + 210.0 * out[4] + 252.0 * out[5]
                 + 210.0 * out[6] + 120.0 * out[7] + 45.0 * out[8]
                 + 10.0 * out[9] + 1.0 * out[10]) / 1024.0
        sm[-6] = (1.0 * out[-11] + 10.0 * out[-10] + 45.0 * out[-9]
                  + 120.0 * out[-8] + 210.0 * out[-7] + 252.0 * out[-6]
                  + 210.0 * out[-5] + 120.0 * out[-4] + 45.0 * out[-3]
                  + 10.0 * out[-2] + 1.0 * out[-1]) / 1024.0
        out = sm
    elif n >= 11:
        sm = out.copy()
        sm[5:-5] = (1.0 * out[:-10] + 10.0 * out[1:-9] + 45.0 * out[2:-8]
                    + 120.0 * out[3:-7] + 210.0 * out[4:-6] + 252.0 * out[5:-5]
                    + 210.0 * out[6:-4] + 120.0 * out[7:-3] + 45.0 * out[8:-2]
                    + 10.0 * out[9:-1] + 1.0 * out[10:]) / 1024.0
        sm[1] = 0.25 * out[0] + 0.5 * out[1] + 0.25 * out[2]
        sm[-2] = 0.25 * out[-3] + 0.5 * out[-2] + 0.25 * out[-1]
        sm[2] = (1.0 * out[0] + 4.0 * out[1] + 6.0 * out[2]
                 + 4.0 * out[3] + 1.0 * out[4]) / 16.0
        sm[-3] = (1.0 * out[-5] + 4.0 * out[-4] + 6.0 * out[-3]
                  + 4.0 * out[-2] + 1.0 * out[-1]) / 16.0
        sm[3] = (1.0 * out[0] + 6.0 * out[1] + 15.0 * out[2]
                 + 20.0 * out[3] + 15.0 * out[4] + 6.0 * out[5]
                 + 1.0 * out[6]) / 64.0
        sm[-4] = (1.0 * out[-7] + 6.0 * out[-6] + 15.0 * out[-5]
                  + 20.0 * out[-4] + 15.0 * out[-3] + 6.0 * out[-2]
                  + 1.0 * out[-1]) / 64.0
        out = sm
    elif n >= 9:
        sm = out.copy()
        sm[4:-4] = (1.0 * out[:-8] + 8.0 * out[1:-7] + 28.0 * out[2:-6]
                    + 56.0 * out[3:-5] + 70.0 * out[4:-4]
                    + 56.0 * out[5:-3] + 28.0 * out[6:-2]
                    + 8.0 * out[7:-1] + 1.0 * out[8:]) / 256.0
        sm[1] = 0.25 * out[0] + 0.5 * out[1] + 0.25 * out[2]
        sm[-2] = 0.25 * out[-3] + 0.5 * out[-2] + 0.25 * out[-1]
        sm[2] = (1.0 * out[0] + 4.0 * out[1] + 6.0 * out[2]
                 + 4.0 * out[3] + 1.0 * out[4]) / 16.0
        sm[-3] = (1.0 * out[-5] + 4.0 * out[-4] + 6.0 * out[-3]
                  + 4.0 * out[-2] + 1.0 * out[-1]) / 16.0
        out = sm
    elif n >= 7:
        sm = out.copy()
        sm[3:-3] = (1.0 * out[:-6] + 6.0 * out[1:-5] + 15.0 * out[2:-4]
                    + 20.0 * out[3:-3] + 15.0 * out[4:-2]
                    + 6.0 * out[5:-1] + 1.0 * out[6:]) / 64.0
        sm[1] = 0.25 * out[0] + 0.5 * out[1] + 0.25 * out[2]
        sm[-2] = 0.25 * out[-3] + 0.5 * out[-2] + 0.25 * out[-1]
        out = sm
    elif n >= 5:
        sm = out.copy()
        sm[2:-2] = (1.0 * out[:-4] + 4.0 * out[1:-3] + 6.0 * out[2:-2]
                    + 4.0 * out[3:-1] + 1.0 * out[4:]) / 16.0
        sm[1] = 0.25 * out[0] + 0.5 * out[1] + 0.25 * out[2]
        sm[-2] = 0.25 * out[-3] + 0.5 * out[-2] + 0.25 * out[-1]
        out = sm
    elif n >= 3:
        sm = out.copy()
        sm[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = sm

    # 分段收缩：低功率段占比最高(~70%)，历史表现为系统性高估，加强收缩；
    # 中段轻微收缩；高段基本保持，避免漏报。
    r2 = out / cap

    low_mask = r2 < 0.2
    if np.any(low_mask):
        # 更强低段收缩：0.872 + 0.128*x  (x in [0,1])
        shrink = 0.872 + 0.128 * (r2[low_mask] / 0.2)
        out[low_mask] = out[low_mask] * shrink

    mid_mask = (r2 >= 0.2) & (r2 < 0.6)
    if np.any(mid_mask):
        out[mid_mask] = out[mid_mask] * 0.9705

    high_mask = r2 >= 0.6
    if np.any(high_mask):
        out[high_mask] = out[high_mask] * 0.9845

    out = np.clip(out, 0.0, cap)

    if np.allclose(out, out[0]):
        out = pp.copy()

    return out