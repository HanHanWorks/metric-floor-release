# -*- coding: utf-8 -*-
"""
P3-C1/C2：机制本体（12 类）+ 从已接受代码中抽取
输出：mechanism_matrix.csv（run × 机制 0/1）、mechanism_stats.csv
"""
import os, re, glob, json
import numpy as np
import pandas as pd

RES = os.environ.get("P3_RESULTS", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results"))
OUT = RES
os.makedirs(OUT, exist_ok=True)

# ---- 12 类机制本体：名称、判据（正则）、中文说明 ----
# 每条正则在代码里命中即视为使用该机制
ONTOLOGY = {
    "M01_global_scale": (
        [r"\bpp\s*\*\s*[\d.]+", r"\bx\s*\*\s*[\d.]+", r"\*\s*1\.[0-2]",
         r"\*\s*0\.(8|85|9|95)", r"scale\s*=", r"out\s*=\s*pp\s*\*\s*[\d.]"],
        "全局缩放：整体乘一个常数，修正系统性高估/低估"),
    "M02_smoothing": (
        [r"rolling\(\s*\d+[^)]*\)\.mean", r"convolve", r"0\.25\s*\*\s*x\[:-2\]",
         r"sm\[1:-1\]", r"\.ewm\(", r"savgol", r"weights\s*=\s*\["],
        "平滑：滑动平均/加权三点/卷积，抑制高频抖动"),
    "M03_median_filter": (
        [r"rolling\([^)]*\)\.median", r"np\.median\(\s*np\.stack", r"medfilt",
         r"median_filter", r"\bmed\s*=\s*np\.median"],
        "中值滤波：用局部中位数替代，抗孤立尖峰"),
    "M04_spike_removal": (
        [r"spike", r"dev\s*>\s*thr", r"abs\(pp\s*-\s*med\)", r"outlier"],
        "尖峰剔除：与局部中位数偏差超阈值即替换"),
    "M05_floor": (
        [r"np\.where\([^)]*<\s*0?\.\d+\s*\*\s*cap", r"floor", r"x\[\s*mask_low\s*\]\s*=\s*[\d.]+\s*\*\s*cap",
         r"clip\(\s*[a-z_]*\s*,\s*0?\.[\d]+\s*\*\s*cap"],
        "抬底：低于阈值置为阈值，消除零值与极低出力"),
    "M06_low_shrink": (
        [r"mask_low[\s\S]{0,60}\*\s*0\.[0-9]+", r"low_thr", r"\*\s*0\.(7|8|85|9)[\s;)]",
         r"shrink"],
        "低出力收缩：低段按比例压向零，压低相对误差"),
    "M07_clip_capacity": (
        [r"clip\(\s*[^,]+,\s*0\s*,\s*[^)]*cap", r"\*?\s*0\.9\d?\s*\*\s*cap", r"np\.minimum\(",
         r"cap\s*\)\s*$"],
        "容量约束/削顶：输出限制在 [0, k·Cap]"),
    "M08_ramp_limit": (
        [r"np\.diff\(", r"ramp", r"max_up", r"max_down", r"cumsum\(", r"np\.maximum\.accumulate"],
        "爬坡限制：相邻点变化幅度受约束"),
    "M09_lag_blend": (
        [r"shift\(\s*9[0-9]", r"np\.roll\(", r"blend", r"yesterday", r"lag"],
        "滞后混合：与昨日同时刻/前序点按比例混合"),
    "M10_quantile_stretch": (
        [r"percentile\(", r"quantile\(", r"rankdata", r"argsort", r"np\.sort\("],
        "分位拉伸：按分位点重映射动态范围"),
    "M11_calendar_correction": (
        [r"groupby\(\s*['\"]?hour", r"\bhour\b", r"\bdow\b", r"dayofweek", r"diurnal",
         r"time_of_day"],
        "日历修正：按小时/星期分组建模修正"),
    "M12_piecewise": (
        [r"np\.where\([^)]*np\.where", r"mask_mid", r"mask_high", r"piecewise",
         r"seg_", r"\bbins?\b"],
        "分段映射：按多个阈值分档施加不同修正"),
}


def extract(code):
    """返回该代码命中的机制集合（名字列表）"""
    hit = []
    for name, (pats, _) in ONTOLOGY.items():
        for p in pats:
            if re.search(p, code, flags=re.I | re.M):
                hit.append(name)
                break
    return hit


def collect_files():
    """收集所有 run 的最终代码，并按 run 标识归组"""
    rows = []
    for fp in sorted(glob.glob(RES + "/best_*.py")):
        arm = os.path.basename(fp)[4:-3]           # best_E2_R1.py -> E2_R1
        code = open(fp, encoding="utf-8", errors="ignore").read()
        mech = extract(code)
        rows.append(dict(run=arm, arm=arm.split("_s")[0],
                         seed=arm.split("_s")[1] if "_s" in arm else "0",
                         n_char=len(code), n_mech=len(mech),
                         mechs=";".join(mech)))
    return pd.DataFrame(rows)


def main():
    print("=" * 96)
    print("P3-C1｜机制本体（12 类）")
    print("=" * 96)
    for k, (pats, desc) in ONTOLOGY.items():
        print("  %-24s %s" % (k, desc))
    print("  判据: 每条机制 %d 个正则模式，命中任一即算" %
          int(np.mean([len(v[0]) for v in ONTOLOGY.values()])))

    D = collect_files()
    print("\n" + "=" * 96)
    print("P3-C2｜从 %d 份已接受代码中抽取" % len(D))
    print("=" * 96)
    D.to_csv(OUT + "/mechanism_matrix.csv", index=False, encoding="utf-8-sig")

    allm = [m for s in D.mechs for m in s.split(";") if m]
    cnt = pd.Series(allm).value_counts()
    print("\n  各机制的出现次数（共 %d 次命中，平均每份代码 %.1f 个机制）"
          % (len(allm), D.n_mech.mean()))
    for k in ONTOLOGY:
        print("    %-24s %3d 次  (%.0f%% 的代码)" % (k, cnt.get(k, 0), 100 * cnt.get(k, 0) / len(D)))

    # 按实验臂聚合
    print("\n  按实验臂（每个臂 10 个种子）的机制构成:")
    piv = pd.DataFrame(0, index=sorted(D.arm.unique()), columns=list(ONTOLOGY))
    for _, r in D.iterrows():
        for m in r.mechs.split(";"):
            if m:
                piv.loc[r.arm, m] += 1
    print(piv.to_string())
    piv.to_csv(OUT + "/mechanism_by_arm.csv", encoding="utf-8-sig")

    # 机制共现
    co = pd.DataFrame(0, index=list(ONTOLOGY), columns=list(ONTOLOGY))
    for _, r in D.iterrows():
        ms = [m for m in r.mechs.split(";") if m]
        for a in ms:
            for b in ms:
                co.loc[a, b] += 1
    co.to_csv(OUT + "/mechanism_cooccur.csv", encoding="utf-8-sig")

    print("\n  代码长度: 中位 %d 字符，机制数: 中位 %.1f，范围 %d–%d"
          % (D.n_char.median(), D.n_mech.median(), D.n_mech.min(), D.n_mech.max()))
    print("\n已保存 mechanism_matrix.csv / mechanism_by_arm.csv / mechanism_cooccur.csv")


if __name__ == "__main__":
    main()
