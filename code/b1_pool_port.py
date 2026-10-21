# -*- coding: utf-8 -*-
"""补充: 池内B1组合(各止损档) vs 池内基准 — 用户实际使用场景(选股池内选B1)"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
STOPS = [None, 8, 12, 15, 20]
ROUND_TRIP = (0.00025*2 + 0.0005 + 0.001*2)*PCT

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
kl = {code: [(ft["date"][i].replace("-", ""), float(ft["o"][i]), float(ft["h"][i]), float(ft["l"][i]), float(ft["c"][i]))
             for i in range(len(ft["date"]))] for code, ft in feats.items() if code in pool_codes}

def simulate(arr, sd, stop_pct, hold=20):
    idx = None
    for i, row in enumerate(arr):
        if row[0] > sd: idx = i; break
    if idx is None: return None
    entry = arr[idx][1]
    if not entry or entry <= 0: return None
    sp = entry*(1-stop_pct/100.0) if stop_pct else None
    peak = entry; exit_p = None; last = min(idx+hold-1, len(arr)-1)
    for j in range(idx, last+1):
        _, o, h, l, c = arr[j]
        if sp is not None and l is not None and l <= sp: exit_p = sp; break
        if c:
            if c > peak: peak = c
            if j > idx and peak and c <= peak*0.5: exit_p = c; break
        if j == last: exit_p = c if c else exit_p
    if exit_p is None or not exit_p: return None
    return (exit_p/entry-1)*PCT

def port_metrics(monthly):
    yms = sorted(monthly); rs = [monthly[y] for y in yms]; n = len(rs)
    if n < 2 or st.pstdev(rs) == 0: return None
    eq, cur = [], 1.0
    for r in rs: cur *= (1+r/100.0); eq.append(cur)
    total = cur; ann = (total**(12.0/n)-1)*100
    mu, sd_ = st.mean(rs), st.pstdev(rs)
    peak, mdd = eq[0], 0.0
    for v in eq:
        if v > peak: peak = v
        dd = v/peak-1
        if dd < mdd: mdd = dd
    return {"n_months": n, "total": (total-1)*100, "ann": ann, "sharpe": mu/sd_*math.sqrt(12), "maxdd": mdd*100}

sigs = []
for code, ft in feats.items():
    if code not in pool_codes: continue
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = ft["date"][i].replace("-", "")
        if D0 <= ds <= D1: sigs.append((code, ds))
print("池内信号: %d" % len(sigs))

out = {}
for sp in STOPS:
    label = "不止损" if sp is None else "止损%d%%" % sp
    monthly = {}
    for code, ds in sigs:
        pnl = simulate(kl[code], ds, sp)
        if pnl is not None: monthly.setdefault(ds[:6], []).append(pnl)
    a = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    per = {}
    for pname in ["训练期", "验证期"]:
        m = {ym: st.mean(v) for ym, v in monthly.items() if ("20250908" > ym+"01") == (pname == "训练期")}
        pm = port_metrics(m)
        if pm: per[pname] = pm
    monthly_c = {ym: st.mean([v-ROUND_TRIP for v in vs]) for ym, vs in monthly.items()}
    ac = port_metrics(monthly_c)
    out[label] = {"all": a, "period": per, "costed": ac}
    print("%-8s 累计%+7.1f%% 年化%+6.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练%+7.1f%% 验证%+7.1f%% | 计费%+7.1f%%" % (
        label, a["total"], a["ann"], a["sharpe"], a["maxdd"],
        per["训练期"]["total"], per["验证期"]["total"], ac["total"]))

# 池内基准
monthly = {}
for code, ft in feats.items():
    if code not in pool_codes: continue
    t1 = ft["t1_20"]; dates = ft["date"]
    for i in range(59, len(t1)-20):
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if D0 <= ds <= D1: monthly.setdefault(ds[:6], []).append(float(t1[i])*PCT)
b = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
per = {}
for pname in ["训练期", "验证期"]:
    m = {ym: st.mean(v) for ym, v in monthly.items() if ("20250908" > ym+"01") == (pname == "训练期")}
    pm = port_metrics(m)
    if pm: per[pname] = pm
out["池内基准(等权持有)"] = {"all": b, "period": per}
print("池内基准   累计%+7.1f%% 年化%+6.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练%+7.1f%% 验证%+7.1f%%" % (
    b["total"], b["ann"], b["sharpe"], b["maxdd"], per["训练期"]["total"], per["验证期"]["total"]))

json.dump(out, open(os.path.join(HERE, "b1_pool_port.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> b1_pool_port.json")
