# -*- coding: utf-8 -*-
"""公共层（带缓存）：数据加载 + 四层目标函数 + 官方ACC"""
import os, pickle
import numpy as np
import pandas as pd

R = os.environ.get("WIND_DATA_ROOT", "")
S = R + "/2原始数据/3Second_data"
OUT = os.environ.get("P3_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"))
CACHE = os.environ.get("P3_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".cache"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def _load_raw(year=2024):
    if not R:
        raise SystemExit("WIND_DATA_ROOT is not set: point it at the directory holding the\n"
                         "restricted metered and forecast series (see data/README.md).")
    cap = pd.read_csv(R + "/4处理后数据/filtered_56_farm.csv")
    cap.columns = [c.strip().lstrip('\ufeff') for c in cap.columns]
    cap['farm_id'] = cap['farm_id'].astype(str).str.strip()
    cap['Cap'] = pd.to_numeric(cap['capability'].astype(str).str.strip(" '"), errors='coerce')
    cap['nm'] = cap['farm_name'].astype(str).str.strip(" '")
    cap = cap[(cap.Cap > 10) & cap.nm.str.contains('风电')]

    real = pd.read_csv(S + "/75-电站实际功率%d-bs_data_real_fifteen_%d.csv" % (year, year),
                       usecols=['farm_id', 'data_time', 'real_power'],
                       dtype={'farm_id': str}, parse_dates=['data_time'])
    real['farm_id'] = real['farm_id'].str.strip()
    fc = pd.read_csv(S + "/81-短期预测功率表%d-bs_data_short_power_%d.csv" % (year, year),
                     usecols=['farm_id', 'data_time', 'predict_power', 'data_source'],
                     dtype={'farm_id': str, 'data_source': str}, parse_dates=['data_time'])
    fc = fc[fc.data_source.str.strip() == '1'][['farm_id', 'data_time', 'predict_power']]
    fc['farm_id'] = fc['farm_id'].str.strip()
    fc = fc.drop_duplicates(['farm_id', 'data_time'])

    m = real.merge(fc, on=['farm_id', 'data_time'], how='inner').merge(
        cap[['farm_id', 'Cap', 'nm']], on='farm_id', how='inner')
    m['PP'] = pd.to_numeric(m.predict_power, errors='coerce')
    m['PM'] = pd.to_numeric(m.real_power, errors='coerce')
    m = m.dropna(subset=['PM', 'PP'])
    m = m[m.PM >= 0]
    return m


def load_station_data(year=2024, n_station=4, seed=0, min_days=200, start=None, end=None):
    """返回 {farm_id: DataFrame[dt, PM, PP, Cap, nm]}，带磁盘缓存
    start/end: 可选日期窗口（'2023-01-01' 形式），用于滑窗战役"""
    wtag = "" if start is None else "_%s_%s" % (start.replace("-", ""), end.replace("-", ""))
    cp = CACHE + "/stations_y%d_n%d_s%d_d%d%s.pkl" % (year, n_station, seed, min_days, wtag)
    if os.path.exists(cp):
        try:
            return pickle.load(open(cp, "rb"))
        except Exception:
            pass
    m = _load_raw(year)
    if start is not None:
        m = m[(m.data_time >= start) & (m.data_time < end)]
    ids = sorted(m.farm_id.unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    out = {}
    for fid in ids:
        d = m[m.farm_id == fid].sort_values('data_time').reset_index(drop=True)
        if len(d) < 96 * min_days:
            continue
        d = d.rename(columns={'data_time': 'dt'})
        d['dayc'] = d.dt.dt.normalize().astype('int64') // 10**9
        out[fid] = d[['dt', 'dayc', 'PM', 'PP', 'Cap', 'nm']]
        if len(out) >= n_station:
            break
    pickle.dump(out, open(cp, "wb"))
    return out


# ---------------- 四层目标 ----------------
def acc_official(PM, PP, Cap):
    """细则官方口径：相对误差 + 0.2Cap 下限"""
    d = np.where(PM >= 0.2 * Cap, PM, 0.2 * Cap)
    d = np.where(d == 0, np.nan, d)
    e = (PM - PP) / d
    e = e[~np.isnan(e)]
    if len(e) < 90:
        return np.nan
    return (1 - np.sqrt((e ** 2).mean())) * 100.0


def mase(PM, PP, train_scale):
    if len(PM) < 90:
        return np.nan
    return float(np.abs(PM - PP).mean() / train_scale)


def penalty_MWh(PM, PP, Cap, theta=60.0, k=0.2):
    a = acc_official(PM, PP, Cap)
    if np.isnan(a):
        return np.nan
    return float(Cap * k * max(0.0, theta - a))


def daily_metrics(df, pred, train_scale):
    """向量化：按自然日聚合四层指标（比 groupby 快约 20 倍）"""
    PM = df.PM.values.astype(float)
    PP = np.asarray(pred, dtype=float).ravel()
    C = float(df.Cap.iloc[0])
    dc = df.dayc.values.astype(np.int64)

    d = np.where(PM >= 0.2 * C, PM, 0.2 * C)
    d = np.where(d == 0, np.nan, d)
    e2 = ((PM - PP) / d) ** 2
    ae = np.abs(PM - PP)
    se = (PM - PP) ** 2

    uk, inv = np.unique(dc, return_inverse=True)
    n = np.bincount(inv, minlength=len(uk)).astype(float)
    s_e2 = np.bincount(inv, weights=np.nan_to_num(e2, nan=0.0), minlength=len(uk))
    s_ae = np.bincount(inv, weights=ae, minlength=len(uk))
    s_se = np.bincount(inv, weights=se, minlength=len(uk))

    ok = n >= 90
    if ok.sum() == 0:
        return pd.DataFrame(columns=["mse", "mase", "acc", "pen"])
    acc = (1 - np.sqrt(s_e2[ok] / n[ok])) * 100.0
    return pd.DataFrame(dict(
        mse=s_se[ok] / n[ok],
        mase=s_ae[ok] / n[ok] / train_scale,
        acc=acc,
        pen=C * 0.2 * np.maximum(0.0, 60.0 - acc),
    ))


def summarize(mt):
    return dict(mse=float(mt.mse.mean()), mase=float(mt.mase.mean()),
                acc=float(mt.acc.mean()), pen=float(mt.pen.mean()),
                below60=float((mt.acc < 60).mean() * 100))


def data_profile(df):
    """只含预测侧信息的统计画像，供 agent 决策（绝不含 PM）"""
    PP = df.PP.values.astype(float)
    C = float(df.Cap.iloc[0])
    r = PP / C
    ramp = np.abs(np.diff(PP)) / C
    return dict(
        n=len(PP), cap=round(C, 1),
        pp_mean_ratio=round(float(r.mean()), 3),
        pp_p50=round(float(np.percentile(r, 50)), 3),
        pp_p90=round(float(np.percentile(r, 90)), 3),
        pp_p99=round(float(np.percentile(r, 99)), 3),
        frac_below_02cap=round(float((r < 0.2).mean()), 3),
        frac_below_01cap=round(float((r < 0.1).mean()), 3),
        frac_zero=round(float((PP <= 0.001 * C).mean()), 3),
        frac_above_09cap=round(float((r > 0.9).mean()), 3),
        ramp_p50=round(float(np.percentile(ramp, 50)), 4),
        ramp_p95=round(float(np.percentile(ramp, 95)), 4),
        ramp_p99=round(float(np.percentile(ramp, 99)), 4),
        ramp_max=round(float(ramp.max()), 4),
        day_autocorr=round(float(pd.Series(PP).autocorr(96)), 3),
        step_autocorr=round(float(pd.Series(PP).autocorr(1)), 3),
    )
