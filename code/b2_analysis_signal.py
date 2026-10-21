# -*- coding: utf-8 -*-
"""
B2因子重构评估 — Step2 信号层检验
口径与B1一致: 信号日(T+1确认日)收盘后可知 → 次日开盘入场 → 20个交易日
核心归因: B1候选直接进 vs B2确认后进(配对); 候选中被确认/未被确认子集的前瞻差异
"""
import pickle, os, json, math, statistics as st, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d2 = pickle.load(open(os.path.join(HERE, "b2_features.pkl"), "rb"))
feats = d2["feats"]
d1 = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats1 = d1["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    srt = np.sort(v); top5 = srt[int(len(srt)*0.95):]
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std()),
            "top5_share": float(100*top5.sum()/v.sum()) if v.sum() != 0 else None}

R = {}

# ---------- 1. B2信号表 ----------
rows = []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["b2_sig"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if np.isnan(ft["t1_20"][i]): continue
        rows.append({"code": code, "date": ds, "strong": bool(ft["b2_strong"][i]),
                     "score": float(ft["b2_score"][i]), "in_pool": 1 if code in pool_codes else 0,
                     "t1_20": float(ft["t1_20"][i])*PCT, "fwd20": float(ft["fwd20"][i])*PCT,
                     "j": float(ft["j"][i]), "vr5": float(ft["vr5"][i]), "chg": float(ft["chg"][i]),
                     "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
                     "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
                     "upper_shadow": float(ft["upper_shadow"][i]),
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
                     "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
                     "vol_ratio": float(ft["vol_ratio"][i]),
                     "cand_score": float(ft["b1_cand_score"][i-1]) if i >= 1 else np.nan})
print("B2信号(完整前瞻): %d (强 %d / 普通 %d)" % (len(rows), sum(r["strong"] for r in rows), sum(not r["strong"] for r in rows)))

groups = {"全体": rows,
          "训练期": [r for r in rows if period_of(r["date"]) == "训练期"],
          "验证期": [r for r in rows if period_of(r["date"]) == "验证期"],
          "强信号(score>=7)": [r for r in rows if r["strong"]],
          "普通信号(score5-6)": [r for r in rows if not r["strong"]],
          "池内": [r for r in rows if r["in_pool"]],
          "池外": [r for r in rows if not r["in_pool"]],
          "池内-训练": [r for r in rows if r["in_pool"] and period_of(r["date"]) == "训练期"],
          "池内-验证": [r for r in rows if r["in_pool"] and period_of(r["date"]) == "验证期"],
          "池外-训练": [r for r in rows if not r["in_pool"] and period_of(r["date"]) == "训练期"],
          "池外-验证": [r for r in rows if not r["in_pool"] and period_of(r["date"]) == "验证期"]}
R["groups"] = {k: desc([r["t1_20"] for r in v]) for k, v in groups.items()}
for k in ["全体", "训练期", "验证期", "强信号(score>=7)", "普通信号(score5-6)", "池内", "池外", "池内-训练", "池内-验证", "池外-训练", "池外-验证"]:
    v = R["groups"][k]
    if v: print("%-16s n=%6d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%% p10=%+7.1f p90=%+7.1f" % (k, v["n"], v["mean"], v["median"], v["win"], v["p10"], v["p90"]))

# ---------- 2. 核心归因: B1候选 vs B2确认(配对) ----------
# 路径A(候选日di入场): o[di+1] -> c[di+20]  = t1_20[di]
# 路径B(B2日j=di+1入场): o[j+1] -> c[j+20] = t1_20[j]
pairA, pairB = [], []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["b1_cand"])[0]:
        if i < 1 or i+1 >= len(ft["t1_20"]): continue
        ds_c = dates[i].replace("-", ""); ds_b = dates[i+1].replace("-", "")
        if not (D0 <= ds_c <= D1): continue
        a = ft["t1_20"][i]; b = ft["t1_20"][i+1]
        if np.isnan(a) or np.isnan(b): continue
        pairA.append(float(a)*PCT); pairB.append(float(b)*PCT)
R["paired"] = {"A_B1候选直接进": desc(pairA), "B_B2确认后进": desc(pairB)}
print("\n配对归因(同一事件集, n=%d):" % len(pairA))
print("  A B1候选直接进(T+1开盘): n=%d mean=%+.2f%% median=%+.2f%% win=%.1f%%" % (
    len(pairA), np.mean(pairA), np.median(pairA), 100*np.mean(np.array(pairA) > 0)))
print("  B B2确认后进(T+2开盘):   n=%d mean=%+.2f%% median=%+.2f%% win=%.1f%%" % (
    len(pairB), np.mean(pairB), np.median(pairB), 100*np.mean(np.array(pairB) > 0)))

# 候选集中: 被确认 vs 未被确认(从候选日入场)
conf, unconf = [], []
# 关键补充: 未确认候选同样在 T+2 开盘入场(o[di+2]->c[di+21]) — 与B2同一入场时点的干净对照
unconf_t2 = []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]; c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    n = len(t1)
    for i in np.where(ft["b1_cand"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if np.isnan(t1[i]): continue
        confirmed = (i+1 < n and bool(ft["b2_sig"][i+1]))
        if confirmed: conf.append(float(t1[i])*PCT)
        else:
            unconf.append(float(t1[i])*PCT)
            # T+2开盘入场: o[i+2] -> c[i+21]
            if i+21 < n and o[i+2] > 0:
                unconf_t2.append(float(c[i+21]/o[i+2]-1)*PCT)
# B2集合的 T+2 入场即 t1_20[j], j=di+1 -> 已在 rows
R["cand_split"] = {"被B2确认": desc(conf), "未被确认": desc(unconf), "未被确认_T2入场": desc(unconf_t2)}
print("\nB1候选集内部(从候选日T+1开盘入场):")
print("  被B2确认(次日):   n=%d mean=%+.2f%% win=%.1f%%" % (len(conf), np.mean(conf), 100*np.mean(np.array(conf) > 0)))
print("  未被确认:          n=%d mean=%+.2f%% win=%.1f%%" % (len(unconf), np.mean(unconf), 100*np.mean(np.array(unconf) > 0)))
print("  未被确认(改T+2入场): n=%d mean=%+.2f%% win=%.1f%%  <-- 与B2同入场时点" % (
    len(unconf_t2), np.mean(unconf_t2), 100*np.mean(np.array(unconf_t2) > 0)))

# ---------- 3. 十分位 ----------
allv = np.sort(np.array([r["t1_20"] for r in rows]))
dec = {}
for i in range(10):
    lo, hi = np.percentile(allv, i*10), np.percentile(allv, (i+1)*10)
    seg = allv[(allv >= lo) & (allv <= hi)] if i < 9 else allv[allv >= lo]
    dec["D%d(%.1f~%.1f%%)" % (i+1, lo, hi)] = {"n": int(len(seg)), "share": float(100*len(seg)/len(allv)),
                                               "pnl_share": float(100*seg.sum()/allv.sum()), "mean": float(seg.mean())}
R["deciles"] = dec
print("\n十分位:")
for k, v in dec.items(): print("  %-18s 数量%5.1f%% 盈亏%+7.1f%% 均值%+7.2f%%" % (k, v["share"], v["pnl_share"], v["mean"]))

# ---------- 4. IC ----------
daily = collections.defaultdict(lambda: {"sig": [], "score": [], "fwd": [], "sigfwd": [], "nosigfwd": []})
for code, ft in feats.items():
    dates = ft["date"]; sb = ft["b2_sig"]; sc = ft["b2_score"]; fw = ft["fwd20"]
    for i in range(119, len(dates)-20):
        if np.isnan(fw[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        dd = daily[ds]
        dd["fwd"].append(fw[i]*PCT)
        dd["score"].append(float(sc[i]) if not np.isnan(sc[i]) else np.nan)
        if sb[i]: dd["sig"].append(1.0); dd["sigfwd"].append(fw[i]*PCT)
        else: dd["sig"].append(0.0); dd["nosigfwd"].append(fw[i]*PCT)

def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); i = 0; sa = a[order]
    while i < len(a):
        j2 = i
        while j2+1 < len(a) and sa[j2+1] == sa[i]: j2 += 1
        ranks[order[i:j2+1]] = (i+j2)/2.0+1; i = j2+1
    return ranks
def spear(x, y):
    if len(x) < 30: return None
    rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else None

ic = {"sig": [], "score": []}
spread = []
for ds in sorted(daily):
    dd = daily[ds]
    for k in ic:
        xs = dd[k]; ys = dd["fwd"]
        m = ~np.isnan(xs)
        r = spear(np.array(xs)[m], np.array(ys)[m])
        if r is not None: ic[k].append(r)
    if dd["sigfwd"] and dd["nosigfwd"]:
        spread.append(np.mean(dd["sigfwd"])-np.mean(dd["nosigfwd"]))
ic_out = {}
for k, v in ic.items():
    v = np.array(v)
    ic_out[k] = {"n_days": len(v), "mean": float(v.mean()), "std": float(v.std()),
                 "icir": float(v.mean()/v.std()) if v.std() > 0 else None,
                 "pos_ratio": float(100*(v > 0).mean()),
                 "t": float(v.mean()/v.std()*math.sqrt(len(v))) if v.std() > 0 else None}
sp = np.array(spread)
ic_out["spread"] = {"n_days": len(sp), "mean": float(sp.mean()), "pos_ratio": float(100*(sp > 0).mean()),
                    "t": float(sp.mean()/sp.std()*math.sqrt(len(sp))) if sp.std() > 0 else None}
R["ic"] = ic_out
print("\nIC:")
for k in ["sig", "score", "spread"]:
    v = ic_out[k]
    print("  %-7s days=%d mean=%+.4f icir=%+.3f pos=%5.1f%% t=%+.2f" % (k, v["n_days"], v["mean"], v.get("icir") or -9, v["pos_ratio"], v.get("t") or -9))

# ---------- 5. 朴素对照 ----------
def gm(cond, key="t1_20"):
    vals = []
    for code, ft in feats.items():
        arr = ft[key]
        for i in range(119, len(arr)):
            if np.isnan(arr[i]): continue
            ds = ft["date"][i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            if cond(ft, i): vals.append(float(arr[i])*PCT)
    return desc(vals)

naive = {}
naive["B2信号(基准)"] = desc([r["t1_20"] for r in rows])
naive["B1候选(J<20+触多空线+score>=5)"] = desc(pairA)
naive["全股票等权基准"] = gm(lambda ft, i: True)
naive["池内全股票等权基准"] = desc([float(ft["t1_20"][i])*PCT for code, ft in feats.items() if code in pool_codes
                                     for i in range(119, len(ft["t1_20"]))
                                     if not np.isnan(ft["t1_20"][i]) and D0 <= ft["date"][i].replace("-", "") <= D1])
# B1信号(b1管道)同窗口
b1v = []
for code, ft in feats1.items():
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1) or np.isnan(ft["t1_20"][i]): continue
        b1v.append(float(ft["t1_20"][i])*PCT)
naive["B1信号(触任意线)"] = desc(b1v)
R["naive"] = naive
print("\n朴素对照:")
for k, v in naive.items():
    if v: print("  %-30s n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (k, v["n"], v["mean"], v["median"], v["win"]))

monthly = collections.Counter(r["date"][:6] for r in rows)
R["monthly_counts"] = dict(monthly)
print("\n月度信号数:", dict(sorted(monthly.items())))

json.dump(R, open(os.path.join(HERE, "b2_signal_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, "b2_signals_flat.csv"), "w", encoding="utf-8", newline="") as f:
    w = __import__("csv").DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nsaved -> b2_signal_results.json / b2_signals_flat.csv")
