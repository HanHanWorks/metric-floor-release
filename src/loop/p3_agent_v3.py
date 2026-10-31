# -*- coding: utf-8 -*-
"""
P3 agent：跨年战役（C3）+ 记忆开关（C4）
与 P1 的引擎差异：
  1) --year 指定数据年份（纵向战役）
  2) --memory 开关：把已确认有效/已被拒的机制注入提示词
  3) --policy none|full|regime：知识库检索策略（C9）
       none   = 全遗忘
       full   = 全保留（不分 regime，只给平均效果）
       regime = 按 regime 索引（给出每个机制在各出力段的效果，提示分段适用）
  3) 记录每次被接受代码的机制集合（trace），用于重发现率与半衰期
"""
import os, sys, json, time, re, argparse
import numpy as np
import pandas as pd
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_core import load_station_data, daily_metrics, summarize, data_profile
from p3_ontology import ONTOLOGY, extract

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


def llm(prompt, max_tokens=4000, temp=0.4, retries=4, think=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "enable_thinking": bool(think)}
    data = json.dumps(body).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, data=data,
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.loads(r.read().decode())
            return (j["choices"][0]["message"].get("content") or "").strip(), j.get("usage", {})
        except Exception as e:
            last = e; time.sleep(2 + 2 * k)
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


def evaluate(stations, code, scale):
    outs = []
    for fid, df in stations.items():
        cap = float(df.Cap.iloc[0])
        pred, err = run_code(code, make_view(df), cap)
        if err:
            return None, err
        mt = daily_metrics(df, pred, scale[fid])
        if len(mt) < 20:
            continue
        outs.append(summarize(mt))
    if not outs:
        return None, "无有效评估"
    return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}, None


REGIMES = ["R_low", "R_mid", "R_high", "R_ramp"]
REGIME_CN = {"R_low": "低出力段(<20%Cap)", "R_mid": "中出力段(20-60%Cap)",
             "R_high": "高出力段(>60%Cap)", "R_ramp": "强爬坡段(|Δ|>10%Cap/15min)"}


def load_kb(path):
    if not path:
        return None
    try:
        return json.load(open(os.path.expanduser(path)))
    except Exception:
        return None


def build_mem(policy, kb, known_good, tried_bad):
    """按策略组装记忆注入文本"""
    if policy == "none" or not kb:
        return ""
    mechs = kb.get("mechanisms", {})
    if policy == "full":
        good = [(k, v) for k, v in mechs.items() if v.get("avg", 0) > 0]
        good.sort(key=lambda x: -x[1]["avg"])
        lines = ["可用机制清单（已验证在历史战役中有效，可直接复用，不必重新推导）:"]
        for k, v in good:
            lines.append("  - %s %s：平均 %+.2f 分" % (k.split("_")[0], v["desc"][:24], v["avg"]))
        lines.append("已被尝试且被拒绝的机制（不要再提）:")
        lines.append("  " + (", ".join(x.split("_")[0] for x in tried_bad) or "（暂无）"))
        return "\n".join(lines)

    if policy == "regime":
        prof = kb.get("target_regime_profile", {})
        lines = ["知识库（按出力段索引）。每个机制只在其有效的出力段内可用："]
        for k, v in mechs.items():
            segs = [(r, v[r]) for r in REGIMES if r in v and abs(v[r]) > 0.05]
            if not segs:
                continue
            segs.sort(key=lambda x: -x[1])
            pos = [REGIME_CN[r] for r, e in segs if e > 0]
            neg = [REGIME_CN[r] for r, e in segs if e < 0]
            parts = []
            if pos:
                parts.append("有效于 " + "、".join(pos))
            if neg:
                parts.append("**会损害** " + "、".join(neg))
            lines.append("  - %s %s：%s" % (k.split("_")[0], v["desc"][:24], "；".join(parts)))
        if prof:
            top = sorted(prof.items(), key=lambda x: -x[1])[:2]
            lines.append("本战役出力段构成：" + "、".join("%s 占 %.0f%%" % (REGIME_CN[r], 100 * f)
                                                      for r, f in top))
        lines.append("提示：对只在部分出力段有效的机制，请用条件语句（如 np.where(pp > 0.6*cap, ...)）"
                     "限定适用范围，不要全局套用。")
        lines.append("已被尝试且被拒绝的机制（不要再提）:")
        lines.append("  " + (", ".join(x.split("_")[0] for x in tried_bad) or "（暂无）"))
        return "\n".join(lines)

    if policy in ("names", "effects", "code", "placebo", "oracle"):
        mechs = kb.get("mechanisms", {})
        items = sorted(mechs.items(), key=lambda x: -x[1].get("avg", 0))
        if policy == "placebo":
            rng = np.random.RandomState(0)
            vals = [v.get("avg", 0.0) for _, v in items]
            rng.shuffle(vals)
            items = [(k, dict(v, avg=vals[i])) for i, (k, v) in enumerate(items)]
        if policy == "oracle":
            keep = set(kb.get("oracle", []))
            items = [(k, v) for k, v in items if k in keep]
            head = "历史最优机制集（已确认在本任务上收益最高，直接照此实现）:"
        else:
            head = "可用机制清单（来自历史战役，可直接复用，不必重新推导）:"
        lines = [head]
        for k, v in items:
            nm = k.split("_")[0]
            if policy == "names":
                lines.append("  - %s %s" % (nm, v.get("desc", "")[:16]))
            elif policy in ("effects", "placebo"):
                lines.append("  - %s %s：平均 %+.2f 分" % (nm, v.get("desc", "")[:16], v.get("avg", 0.0)))
            else:
                lines.append("  - %s %s（平均 %+.2f 分）示例:" % (nm, v.get("desc", "")[:16], v.get("avg", 0.0)))
                for ln in str(v.get("snippet", "")).split("\n"):
                    lines.append("      " + ln)
        txt = "\n".join(lines)
        # 统一 pad 到同一长度，控制"提示长度"这一混淆
        filler = "\n（以上机制来自同一批历史战役，按平均收益降序排列。）"
        while len(txt) < 900:
            txt += filler
        return txt

    # 兼容旧行为
    return MEM_TPL.format(
        good=", ".join("%s(%s)" % (k.split("_")[0], ONTOLOGY[k][1][:10]) for k in known_good) or "（暂无）",
        bad=", ".join("%s" % k.split("_")[0] for k in tried_bad) or "（暂无）")


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

MEM_TPL = """可用机制清单（已在本轮之前被验证有效，可直接复用，不必重新推导）:
{good}
已被尝试且被拒绝的机制（不要再提）:
{bad}"""


def run_arm(year, seed, memory, n_iter=60, n_station=4, policy=None,
            kb=None, start=None, end=None, accept="greedy", tol=0.0,
            cap="base", bestk=1):
    if policy is None:
        policy = "mem" if memory else "nomem"
    wtag = "" if start is None else "_%s" % start.replace("-", "")[2:]
    tag = "Y%d_%s%s_%s_s%d" % (year, policy, wtag, accept if accept != "greedy" else cap, seed)
    print("=" * 70, flush=True)
    print("RUN %s | year=%d policy=%s seed=%d window=%s" %
          (tag, year, policy, seed, "%s~%s" % (start, end) if start else "全年"), flush=True)
    stations = load_station_data(year, n_station, seed,
                                 min_days=80 if (start is not None or year >= 2025) else 200,
                                 start=start, end=end)
    scale = {f: float(np.abs(np.diff(d.PM.values)).mean()) for f, d in stations.items()}
    if not stations:
        print("⚠ 无可用场站，退出", flush=True)
        return
    print("场站: %s" % [d.nm.iloc[0] for d in stations.values()], flush=True)
    profs = [data_profile(d) for d in stations.values()]
    prof = {k: round(float(np.mean([p[k] for p in profs])), 4) for k in profs[0]}
    prof_txt = "  ".join("%s=%s" % (k, v) for k, v in prof.items())

    code = BASE_CODE
    hist, rows, trace = [], [], []
    known_good = []       # 当前被接受代码里的机制
    tried_bad = []        # 被拒候选里出现的机制
    t0 = time.time()

    for it in range(1, n_iter + 1):
        try:
            m, err = evaluate(stations, code, scale)
            if err:
                print("[%02d] 当前代码评估失败 %s" % (it, err), flush=True); break
            cur_mechs = extract(code)
            rows.append(dict(run=tag, year=year, memory=int(memory), policy=policy,
                             start=start or "", seed=seed, it=it,
                             mase=m["mase"], acc=m["acc"], pen=m["pen"],
                             n_mech=len(cur_mechs), mechs=";".join(cur_mechs)))
            print("[%02d] acc=%.2f mase=%.4f 机制=%d %s"
                  % (it, m["acc"], m["mase"], len(cur_mechs), ",".join(x[:3] for x in cur_mechs)), flush=True)

            mem = build_mem(policy, kb, known_good, tried_bad)
            if hist:
                hs = "\n".join("第%d轮: ACC=%.2f MASE=%.4f" % (h["it"], h["acc"], h["mase"]) for h in hist[-6:])
                fb = "历史表现（同一评估集）:\n" + hs
            else:
                fb = "历史表现：（首轮）"
            prompt = PROMPT.format(profile=prof_txt, memory=mem, code=code, fb=fb)

            if cap == "think":
                temp = min(0.9, 0.4 + 0.5 * it / max(1, n_iter))
                txt, usage = llm(prompt, max_tokens=16000, temp=temp, think=True)
            elif cap == "hot":
                txt, usage = llm(prompt, temp=1.2)
            else:
                temp = min(0.9, 0.4 + 0.5 * it / max(1, n_iter))
                txt, usage = llm(prompt, temp=temp)
            if not txt:
                continue

            # best-of-k：采 k 个候选，先按 ACC 取本地最优，再进门
            cands = []
            def parse(t):
                mm = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
                return (mm.group(1) if mm else t).strip()
            cands.append(parse(txt))
            for _ in range(bestk - 1):
                try:
                    t2, _ = llm(prompt, temp=min(1.2, 0.6 + 0.4 * it / max(1, n_iter)), think=(cap == "think"))
                    cands.append(parse(t2))
                except Exception:
                    pass
            new, m2, err2 = None, None, "none"
            for c in cands:
                if "def transform" not in c:
                    continue
                mm_, e_ = evaluate(stations, c, scale)
                if e_:
                    continue
                if m2 is None or mm_["acc"] > m2["acc"]:
                    new, m2, err2 = c, mm_, None
            if err2 or new is None:
                continue
            new_mechs = extract(new)
            hist.append(dict(it=it, **m2))
            ok = (accept == "always") or (m2["acc"] > m["acc"] + (tol if accept == "tol" else 0.0))
            if ok:
                # 记录机制层面的"重发现"：新代码的机制集是否已在 known_good 中
                overlap = set(new_mechs) & set(known_good)
                novel = set(new_mechs) - set(known_good)
                trace.append(dict(run=tag, it=it, accepted=1, acc=m2["acc"],
                                  mechs=";".join(new_mechs),
                                  n_overlap=len(overlap), n_novel=len(novel),
                                  pure_rediscovery=int(len(novel) == 0 and len(overlap) > 0)))
                code = new
                known_good = new_mechs
                print("       ↑ 接受 acc=%.2f (重叠%d 新增%d)" % (m2["acc"], len(overlap), len(novel)), flush=True)
            else:
                for x in new_mechs:
                    if x not in tried_bad:
                        tried_bad.append(x)
                trace.append(dict(run=tag, it=it, accepted=0, acc=m2["acc"],
                                  mechs=";".join(new_mechs), n_overlap=0, n_novel=0,
                                  pure_rediscovery=0))
                print("       × 拒绝 %.2f→%.2f" % (m["acc"], m2["acc"]), flush=True)
            pd.DataFrame(rows).to_csv(OUT + "/trace_%s.csv" % tag, index=False)
            pd.DataFrame(trace).to_csv(OUT + "/steps_%s.csv" % tag, index=False)
        except Exception as e:
            print("[%02d] 异常 %s" % (it, str(e)[:100]), flush=True)
            time.sleep(1); continue

    open(OUT + "/code_%s.py" % tag, "w").write(code)
    meta = dict(run=tag, year=year, memory=int(memory), policy=policy, accept=accept, tol=tol, cap=cap, bestk=bestk,
                start=start, end=end, seed=seed, n_iter=len(rows),
                acc_first=rows[0]["acc"] if rows else None,
                acc_last=rows[-1]["acc"] if rows else None,
                n_accept=int(sum(x["accepted"] for x in trace)),
                n_pure_rediscovery=int(sum(x["pure_rediscovery"] for x in trace)),
                wall_sec=round(time.time() - t0, 1))
    json.dump(meta, open(OUT + "/meta_%s.json" % tag, "w"), indent=2, ensure_ascii=False)
    if rows:
        print("→ %s ΔACC=%+.2f 接受%d次 纯重发现%d次 %.0fs"
              % (tag, rows[-1]["acc"] - rows[0]["acc"], meta["n_accept"],
                 meta["n_pure_rediscovery"], time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory", type=int, default=0)
    ap.add_argument("--policy", type=str, default=None,
                    help="none|full|regime|mem|nomem")
    ap.add_argument("--kb", type=str, default=None, help="知识库 json 路径")
    ap.add_argument("--start", type=str, default=None, help="窗口起点 YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="窗口终点（不含）")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--accept", type=str, default="greedy", choices=["greedy", "always", "tol"])
    ap.add_argument("--tol", type=float, default=0.5, help="accept=tol 时的放宽量")
    ap.add_argument("--cap", type=str, default="base", choices=["base", "think", "hot", "bestk"])
    ap.add_argument("--bestk", type=int, default=1)
    a = ap.parse_args()
    bk = a.bestk if a.cap == "bestk" else 1
    run_arm(a.year, a.seed, bool(a.memory), a.iters, n_station=a.stations,
            policy=a.policy, kb=load_kb(a.kb), start=a.start, end=a.end,
            accept=a.accept, tol=a.tol, cap=a.cap, bestk=bk)
