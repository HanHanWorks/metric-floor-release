# -*- coding: utf-8 -*-
"""
p3_agent_holdout.py —— C17｜诚实选择：留出验证

与 p3_agent.py 的唯一区别：**接受判据用留出集之外的验证集，报告用留出集。**
目的：区分"增益是真的"与"增益是选择效应"（见 10_analysis/selection_null.py 的 C16c）。

三种模式：
  --holdout none     与 p3_agent.py 完全一致（同一集上接受与报告），用于核对基线
  --holdout station  验证集 = 前一半场站，留出集 = 后一半场站（默认，首选）
  --holdout time     验证集 = 每个场站前 60% 的天，留出集 = 后 40% 的天

产物（写到 results/，文件名带 _ho<mode> 后缀，不会覆盖已有结果）：
  trace_<tag>.csv   逐轮：acc_val / acc_test / 当前代码机制
  steps_<tag>.csv   逐候选：accepted / acc_val / acc_test / 机制
  meta_<tag>.json   汇总：acc_val_first/last、acc_test_first/last、n_accept
  code_<tag>.py     最终代码

用法：
  ~/anaconda3/envs/llm_power_ts/bin/python p3_agent_holdout.py \
      --year 2024 --seed 0 --policy none --iters 60 --holdout station
"""
import os
import sys
import json
import time
import re
import argparse
import numpy as np
import pandas as pd
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_core import load_station_data, daily_metrics, summarize, data_profile
from p3_ontology import extract

# --- endpoint and credentials -------------------------------------------
# The study used an OpenAI-compatible chat-completions endpoint. Nothing is
# hard-coded here: export LLM_HOST / LLM_API_KEY / LLM_MODEL before running.
# Re-running the search verbatim also requires the metered and forecast series,
# which are not distributed (see data/README.md).
HOST = os.environ.get("LLM_HOST", "")
URL = "https://%s/compatible-mode/v1/chat/completions" % HOST if HOST else ""
KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash-1017")
OUT = os.environ.get("P3_RESULTS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results"))
os.makedirs(OUT, exist_ok=True)

BASE_CODE = "import numpy as np\ndef transform(view):\n    return view['PP'].values.astype(float)\n"

PROMPT = """你是时序预测管线改进器。你只能看到预测侧信息，**没有真实功率**。

view 的列（pandas DataFrame，15 分钟粒度，96 点/天）：
- dt: 时间戳
- PP: 原始日前预测功率 (MW)
- Cap: 装机容量 (MW)
- hour: 小时(浮点 0-24)
- dow: 星期几(0-6)
- month: 月份(1-12)

数据画像:
{profile}

{memory}

当前代码:
```python
{code}
```

{fb}

本轮目标: 最大化 ACC（按电网考核口径计算的准确率，%）

只输出一个 ```python 代码块。硬性要求：
1) 必须定义 transform(view)
2) 返回长度必须等于 len(view)
3) 输出不得为常数序列
4) 不得访问 view 之外的任何数据"""


def llm(prompt, max_tokens=4000, temp=0.4, retries=4):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "enable_thinking": False}
    data = json.dumps(body).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, data=data,
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.loads(r.read().decode())
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            last = e
            time.sleep(2 + 2 * k)
    raise last


def make_view(df):
    v = df[["dt", "PP", "Cap"]].copy()
    v["hour"] = v.dt.dt.hour + v.dt.dt.minute / 60.0
    v["dow"] = v.dt.dt.dayofweek
    v["month"] = v.dt.dt.month
    return v


def run_code(code, view, cap):
    g = {"np": np, "pd": pd, "len": len, "float": float, "int": int, "abs": abs,
         "min": min, "max": max, "range": range, "sum": sum, "enumerate": enumerate}
    l = {}
    try:
        exec(code, g, l)
        if "transform" not in l:
            return None, "未定义 transform"
        out = np.asarray(l["transform"](view.copy()), dtype=float).ravel()
        if len(out) != len(view):
            return None, "长度不符"
        if not np.isfinite(out).all():
            return None, "含 NaN/Inf"
        if out.std() < 1e-9:
            return None, "常数序列"
        return np.clip(out, 0, cap), None
    except Exception as e:
        return None, "运行错误: %s" % str(e)[:120]


def station_metrics(df, code, scale, cut_day=None):
    """单场站评估。cut_day 给定时只用该日之前的天（做时间留出）。"""
    cap = float(df.Cap.iloc[0])
    pred, err = run_code(code, make_view(df), cap)
    if err:
        return None, err
    d = df
    if cut_day is not None:
        d = df[df.dt < cut_day]
    if len(d) < 96 * 5:
        return None, "有效天数不足"
    # pred 是对全场站算的，按同一时间掩码切片
    mask = (df.dt < cut_day).values if cut_day is not None else np.ones(len(df), bool)
    mt = daily_metrics(d, pred[mask], scale)
    if len(mt) < 5:
        return None, "无有效评估"
    return summarize(mt), None


def evaluate_split(stations, code, scale, mode):
    """返回 (val_metrics, test_metrics, err)。

    none    : val 与 test 相同（复现原行为）
    station : 按 farm_id 排序前后各半
    time    : 每站按时间 60/40 切
    """
    fids = sorted(stations.keys())
    if mode == "station":
        half = max(1, len(fids) // 2)
        groups = {"val": fids[:half], "test": fids[half:]}
        if not groups["test"]:
            groups["test"] = fids[half - 1:]
            groups["val"] = fids[:half - 1]
    else:
        groups = {"val": fids, "test": fids}

    out = {}
    for key, ids in groups.items():
        accs = []
        for fid in ids:
            df = stations[fid]
            cut = None
            if mode == "time":
                days = np.sort(df.dt.dt.normalize().unique())
                if len(days) < 20:
                    return None, None, "天数不足"
                cut = days[int(0.6 * len(days))]
            m, err = station_metrics(df, code, scale[fid], cut if key == "val" else None)
            if err:
                # 留出（后半段）不能靠切前半段实现，单独算
                if mode == "time" and key == "test":
                    m2, err2 = station_metrics(df, code, scale[fid], None)
                    if err2:
                        return None, None, err2
                    m = m2
                else:
                    return None, None, err
            accs.append(m)
        out[key] = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}

    if mode == "time":
        # 时间模式下 val/test 都要重算：val 用前半段，test 用后半段
        accs_val, accs_test = [], []
        for fid in fids:
            df = stations[fid]
            days = np.sort(df.dt.dt.normalize().unique())
            cut = days[int(0.6 * len(days))]
            m_val, e1 = station_metrics(df, code, scale[fid], cut)
            if e1:
                return None, None, e1
            accs_val.append(m_val)
            # 后半段：临时构造切片后的 df 与对应 pred
            cap = float(df.Cap.iloc[0])
            pred, e2 = run_code(code, make_view(df), cap)
            if e2:
                return None, None, e2
            sub = df[df.dt >= cut]
            mt = daily_metrics(sub, pred[(df.dt >= cut).values], scale[fid])
            if len(mt) < 5:
                return None, None, "后半段天数不足"
            accs_test.append(summarize(mt))
        out["val"] = {k: float(np.mean([a[k] for a in accs_val])) for k in accs_val[0]}
        out["test"] = {k: float(np.mean([a[k] for a in accs_test])) for k in accs_test[0]}

    return out["val"], out["test"], None


def run_arm(year, seed, policy="none", n_iter=60, n_station=4, mode="station",
            start=None, end=None, tol=0.0):
    wtag = "" if start is None else "_%s" % start.replace("-", "")[2:]
    tag = "Y%d_%s%s_ho%s_s%d" % (year, policy, wtag, mode, seed)
    print("=" * 70, flush=True)
    print("RUN %s | year=%d policy=%s holdout=%s seed=%d" % (tag, year, policy, mode, seed), flush=True)

    stations = load_station_data(year, n_station, seed,
                                 min_days=80 if (start is not None or year >= 2025) else 200,
                                 start=start, end=end)
    if not stations:
        print("⚠ 无可用场站，退出", flush=True)
        return
    scale = {f: float(np.abs(np.diff(d.PM.values)).mean()) for f, d in stations.items()}
    fids = sorted(stations.keys())
    if mode == "station":
        half = max(1, len(fids) // 2)
        print("验证场站: %s | 留出场站: %s" % (fids[:half], fids[half:]), flush=True)

    profs = [data_profile(d) for d in stations.values()]
    prof = {k: round(float(np.mean([p[k] for p in profs])), 4) for k in profs[0]}
    prof_txt = "  ".join("%s=%s" % (k, v) for k, v in prof.items())

    code = BASE_CODE
    hist, rows, trace = [], [], []
    t0 = time.time()

    for it in range(1, n_iter + 1):
        try:
            v, t, err = evaluate_split(stations, code, scale, mode)
            if err:
                print("[%02d] 当前代码评估失败 %s" % (it, err), flush=True)
                break
            cur_mechs = extract(code)
            rows.append(dict(run=tag, year=year, policy=policy, start=start or "", seed=seed, it=it,
                             acc_val=v["acc"], acc_test=t["acc"],
                             mase_test=t["mase"], n_mech=len(cur_mechs), mechs=";".join(cur_mechs)))
            print("[%02d] acc_val=%.2f acc_test=%.2f 机制=%d" % (it, v["acc"], t["acc"], len(cur_mechs)),
                  flush=True)

            if hist:
                hs = "\n".join("第%d轮: ACC=%.2f" % (h["it"], h["acc"]) for h in hist[-6:])
                fb = "历史表现（同一评估集）:\n" + hs
            else:
                fb = "历史表现：（首轮）"
            prompt = PROMPT.format(profile=prof_txt, memory="", code=code, fb=fb)

            temp = min(0.9, 0.4 + 0.5 * it / max(1, n_iter))
            txt = llm(prompt, temp=temp)
            if not txt:
                continue
            mm = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
            new = (mm.group(1) if mm else txt).strip()
            if "def transform" not in new:
                continue

            v2, t2, err2 = evaluate_split(stations, new, scale, mode)
            if err2:
                continue
            new_mechs = extract(new)
            # ★ 关键：用验证集决定接受，用留出集报告
            if v2["acc"] > v["acc"] + tol:
                hist.append(dict(it=it, acc=v2["acc"]))
                trace.append(dict(run=tag, it=it, accepted=1, acc_val=v2["acc"], acc_test=t2["acc"],
                                  mechs=";".join(new_mechs)))
                code = new
                print("       ↑ 接受 val=%.2f (留出 %.2f)" % (v2["acc"], t2["acc"]), flush=True)
            else:
                trace.append(dict(run=tag, it=it, accepted=0, acc_val=v2["acc"], acc_test=t2["acc"],
                                  mechs=";".join(new_mechs)))
                print("       × 拒绝 val %.2f→%.2f" % (v["acc"], v2["acc"]), flush=True)

            pd.DataFrame(rows).to_csv(OUT + "/trace_%s.csv" % tag, index=False)
            pd.DataFrame(trace).to_csv(OUT + "/steps_%s.csv" % tag, index=False)
        except Exception as e:
            print("[%02d] 异常 %s" % (it, str(e)[:100]), flush=True)
            time.sleep(1)
            continue

    open(OUT + "/code_%s.py" % tag, "w").write(code)
    meta = dict(run=tag, year=year, policy=policy, holdout=mode, start=start, end=end, seed=seed,
                accept_tol=tol,
                n_iter=len(rows),
                acc_val_first=rows[0]["acc_val"] if rows else None,
                acc_val_last=rows[-1]["acc_val"] if rows else None,
                acc_test_first=rows[0]["acc_test"] if rows else None,
                acc_test_last=rows[-1]["acc_test"] if rows else None,
                n_accept=int(sum(x["accepted"] for x in trace)),
                wall_sec=round(time.time() - t0, 1))
    json.dump(meta, open(OUT + "/meta_%s.json" % tag, "w"), indent=2, ensure_ascii=False)
    if rows:
        print("→ %s Δval=%+.2f Δ留出=%+.2f 接受%d次 %.0fs"
              % (tag, rows[-1]["acc_val"] - rows[0]["acc_val"],
                 rows[-1]["acc_test"] - rows[0]["acc_test"], meta["n_accept"], time.time() - t0),
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", type=str, default="none", help="none|full|regime")
    ap.add_argument("--holdout", type=str, default="station", choices=["none", "station", "time"])
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.0, help="接受容差：需超过 v+tol 才接受")
    a = ap.parse_args()
    run_arm(a.year, a.seed, a.policy, a.iters, a.stations, a.holdout, a.start, a.end, a.tol)