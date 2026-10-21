# -*- coding: utf-8 -*-
"""B1报告图表数据: 月度cohort净值序列 + 回撤 + 分布 + 敏感性 + regime + pre60分层"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats, bench = d["feats"], d["bench"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
codes = list(feats.keys())

def simulate(arr, sd, stop_pct, hold=20):
    idx = None
    for i, row in enumerate(arr):
        if row[0] > sd: idx = i; break
    if idx is None: return None
    entry = arr[idx][1]
    if not entry or entry <= 0: return None
    stop_price = entry*(1-stop_pct/100.0) if stop_pct else None
    peak = entry; exit_p = None
    last = min(idx+hold-1, len(arr)-1)
    for j in range(idx, last+1):
        _, o, h, l, c = arr[j]
        if stop_price is not None and l is not None and l <= stop_price:
            exit_p = stop_price; break
        if c:
            if c > peak: peak = c
            if j > idx and peak and c <= peak*0.5:
                exit_p = c; break
        if j == last: exit_p = c if c else exit_p
    if exit_p is None or not exit_p: return None
    return (exit_p/entry-1)*PCT

kl = {code: [(ft["date"][i].replace("-", ""), float(ft["o"][i]), float(ft["h"][i]), float(ft["l"][i]), float(ft["c"][i]))
             for i in range(len(ft["date"]))] for code, ft in feats.items()}

def monthly_series(cond_fn, mode, stop_pct=None):
    monthly = {}
    for code in codes:
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        ft = feats[code]; dates = ft["date"]
        if cond_fn is None:
            t1 = ft["t1_20"]
            for i in range(59, len(t1)-20):
                if np.isnan(t1[i]): continue
                ds = dates[i].replace("-", "")
                if D0 <= ds <= D1: monthly.setdefault(ds[:6], []).append(float(t1[i])*PCT)
        else:
            for i in np.where(ft["sig_base"] > 0)[0]:
                if i < 59: continue
                ds = dates[i].replace("-", "")
                if not (D0 <= ds <= D1): continue
                pnl = simulate(kl[code], ds, stop_pct)
                if pnl is not None: monthly.setdefault(ds[:6], []).append(pnl)
    return {ym: st.mean(v) for ym, v in sorted(monthly.items())}

series = {
    "B1_8pct": monthly_series("sig", "all", 8),
    "B1_nostop": monthly_series("sig", "all", None),
    "eq_all": monthly_series(None, "all"),
    "eq_pool": monthly_series(None, "pool"),
}
# 中证500月度
b = bench["000905"]; dates = b["date"]; cl = b["close"]
m500 = {}; prev = None
for i, ds in enumerate(dates):
    ym = ds.replace("-", "")[:6]
    if prev is not None and ym != prev[0]: m500[ym] = (cl[i]/prev[1]-1)*PCT
    prev = (ym, cl[i])
series["zz500"] = m500

def equity(monthly):
    eq = {}; cur = 1.0
    for ym in sorted(monthly): cur *= (1+monthly[ym]/100.0); eq[ym] = float(cur)
    return eq
out = {"monthly": {k: {ym: float(v) for ym, v in s.items()} for k, s in series.items()},
       "equity": {k: equity(v) for k, v in series.items()},
       "train_end": TRAIN_END}
# 水下曲线(回撤)
for k, eq in out["equity"].items():
    peak = 0.0; dd = {}
    for ym in sorted(eq):
        peak = max(peak, eq[ym]); dd[ym] = (eq[ym]/peak-1)*100
    out.setdefault("drawdown", {})[k] = dd

# 分布直方图数据(t1_20, 2%桶)
allv = []
for code in codes:
    ft = feats[code]; t1 = ft["t1_20"]
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59 or np.isnan(t1[i]): continue
        ds = ft["date"][i].replace("-", "")
        if D0 <= ds <= D1: allv.append(float(t1[i])*PCT)
allv = np.array(allv)
bins = list(range(-60, 61, 4))
hist, edges = np.histogram(allv, bins=bins)
out["hist"] = {"edges": edges[:-1].tolist(), "counts": hist.tolist(),
               "n": len(allv), "mean": float(allv.mean()), "median": float(np.median(allv)),
               "p05": float(np.percentile(allv, 5)), "p95": float(np.percentile(allv, 95))}
json.dump(out, open(os.path.join(HERE, "b1_chart_data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("series months:", {k: len(v) for k, v in series.items()})
print("equity endpoints:", {k: round(list(v.values())[-1], 3) for k, v in out["equity"].items()})
print("hist n=%d mean=%.2f median=%.2f" % (out["hist"]["n"], out["hist"]["mean"], out["hist"]["median"]))
print("saved -> b1_chart_data.json")
