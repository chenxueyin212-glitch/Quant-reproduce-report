# -*- coding: utf-8 -*-
"""
B1微观假设补充 — 量能确认的事前可交易性检验
问题: vol_ratio(后20日/前20日均量) 与 t1_20 共享同一窗口 → 机械同源, 非事前信号
检验: 仅用 T+1~T+5 均量 / 前20日均量 (vol5), 预测:
  (a) 全窗口 t1_20
  (b) 第5日收盘→第20日收盘收益 (c[i+20]/c[i+5]-1) — 剔除共享窗口后的"剩余15日"
若 vol5 高分组在 (b) 上仍显著占优 → 存在可交易的动态确认规则
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats = d["feats"]

rows = []
for code, ft in feats.items():
    dates = ft["date"]; v = ft["v"].astype(np.float64); c = ft["c"].astype(np.float64)
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        n = len(v)
        if i+21 > n-1 or i < 20: continue
        pre = v[i-20:i].mean()
        post5 = v[i+1:i+6].mean()
        if pre <= 0: continue
        vol5 = post5/pre
        ret_full = (c[i+20]/float(ft["o"][i+1])-1)*PCT if ft["o"][i+1] > 0 else np.nan
        ret_after5 = (c[i+20]/c[i+5]-1)*PCT   # 第5日收盘→第20日收盘
        ret_first5 = (c[i+5]/float(ft["o"][i+1])-1)*PCT if ft["o"][i+1] > 0 else np.nan
        rows.append({"vol5": vol5, "ret_full": ret_full, "ret_after5": ret_after5, "ret_first5": ret_first5})
rows = [r for r in rows if not any(np.isnan(r[k]) for k in ["vol5", "ret_full", "ret_after5", "ret_first5"])]
print("样本: %d" % len(rows))

def dstat(v):
    v = np.array(v)
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)), "win": float(100*(v > 0).mean())}

# vol5 五分位
vals = sorted(r["vol5"] for r in rows)
qs = [np.percentile(vals, q) for q in [0, 20, 40, 60, 80, 100]]
out = {"quintiles": {}, "corr": {}}
print("\nvol5(T+1~T+5均量/前20日均量)五分位:")
for i in range(5):
    lo, hi = qs[i], qs[i+1]
    sub = [r for r in rows if lo <= r["vol5"] <= (hi if i == 4 else hi)]
    rf = dstat([r["ret_full"] for r in sub]); ra = dstat([r["ret_after5"] for r in sub]); r5 = dstat([r["ret_first5"] for r in sub])
    out["quintiles"]["Q%d(%.2f~%.2f)" % (i+1, lo, hi)] = {"ret_full": rf, "ret_after5": ra, "ret_first5": r5}
    print("  Q%d(%.2f~%.2f) n=%6d | 全窗口%+6.2f%%(win%5.1f%%) 前5日%+6.2f%% 后15日%+6.2f%%(win%5.1f%%)" % (
        i+1, lo, hi, rf["n"], rf["mean"], rf["win"], r5["mean"], ra["mean"], ra["win"]))

# 相关性(秩相关, 手工)
def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); i = 0; sa = a[order]
    while i < len(a):
        j2 = i
        while j2+1 < len(a) and sa[j2+1] == sa[i]: j2 += 1
        ranks[order[i:j2+1]] = (i+j2)/2.0+1; i = j2+1
    return ranks
def spear(x, y):
    rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
    return float((rx*ry).sum()/math.sqrt((rx**2).sum()*(ry**2).sum()))
x = [r["vol5"] for r in rows]
for k in ["ret_first5", "ret_full", "ret_after5"]:
    y = [r[k] for r in rows]
    r_ = spear(x, y)
    out["corr"][k] = r_
    print("秩相关 vol5 ~ %s: %+.4f" % (k, r_))

# 动态过滤规则回测(简版): 持有至T+20; 若T+5收盘仍低于入场价*(1-x) 则T+6开盘离场
o_arr = {}
for code, ft in feats.items():
    o_arr[code] = ft["o"].astype(np.float64); 
def dyn_sim(code, i, drop_pct):
    ft = feats[code]; o = o_arr[code]; c = ft["c"].astype(np.float64)
    n = len(o)
    if i+21 > n-1: return None
    e = o[i+1]
    if e <= 0: return None
    # T+6开盘离场条件: T+5收盘 <= e*(1-drop)
    if c[i+5] <= e*(1-drop_pct/100.0):
        x = o[i+6] if i+6 < n and o[i+6] > 0 else c[i+5]
        return (x/e-1)*PCT
    return (c[i+20]/e-1)*PCT
sigs = []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if D0 <= ds <= D1: sigs.append((code, int(i)))
dyn = {}
for dp in [0, 3, 5, 8]:
    pnls = []
    for code, i in sigs:
        r = dyn_sim(code, i, dp)
        if r is not None: pnls.append(r)
    s = dstat(pnls)
    dyn["T+5收盘跌破入场价%d%%则T+6离场" % dp] = s
    print("动态规则[%s]: n=%d mean=%+.2f%% median=%+.2f%% win=%.1f%%" % (dp, s["n"], s["mean"], s["median"], s["win"]))
out["dyn_rule"] = dyn
json.dump(out, open(os.path.join(HERE, "b1_micro_vol5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> b1_micro_vol5.json")
