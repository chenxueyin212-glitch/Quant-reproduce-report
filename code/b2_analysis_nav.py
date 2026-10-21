# -*- coding: utf-8 -*-
"""
B2/B1 真实可投资口径: 日度等权再平衡组合
- 信号: T+1开盘入场, 持有20个交易日, 每日收盘价收益; 入场/出场各扣一半往返成本
- 基准: 等权全A / 等权池内 日收益(每日全持仓再平衡), 成本拖累 0.30%/20日/持仓
- 输出 NAV/年化/夏普/回撤 + 训练/验证切分
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
ROUND_TRIP = (0.00025*2 + 0.0005 + 0.001*2)*PCT   # 0.30% 每往返

d1 = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
d2 = pickle.load(open(os.path.join(HERE, "b2_features.pkl"), "rb"))
feats1, feats2 = d1["feats"], d2["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

# 全宇宙日期轴
all_dates = sorted({ds.replace("-", "") for ft in feats1.values() for ds in ft["date"]})
didx = {d: i for i, d in enumerate(all_dates)}
nd = len(all_dates)

def nav_of(daily_rets):
    """daily_rets: dict date-> pct; 返回 NAV序列与指标"""
    eq, cur = {}, 1.0
    for d in all_dates:
        r = daily_rets.get(d, 0.0)
        cur *= (1+r/100.0); eq[d] = cur
    def metrics(lo, hi):
        seg = [eq[d] for d in all_dates if lo <= d < hi]
        if len(seg) < 20: return None
        rs = np.diff(np.log(seg))
        ann = (seg[-1]/seg[0])**(244.0/len(seg))-1
        vol = float(rs.std()*math.sqrt(244))
        peak = 0.0; mdd = 0.0
        for v in seg:
            if v > peak: peak = v
            dd = v/peak-1
            if dd < mdd: mdd = dd
        return {"total": (seg[-1]/seg[0]-1)*100, "ann": ann*100, "vol": vol*100,
                "sharpe": float(rs.mean()/rs.std()*math.sqrt(244)) if rs.std() > 0 else None, "maxdd": mdd*100}
    out = {"全部": metrics(D0, D1+"z"), "训练期": metrics(D0, TRAIN_END), "验证期": metrics(TRAIN_END, D1+"z")}
    return eq, out

# ---------- 信号组合(日度) ----------
def signal_nav(feats, sigkey, mode="all"):
    pos_ret = {}   # date -> list of daily pct returns
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64); dates = ft["date"]
        n = len(c)
        for i in np.where(ft[sigkey] > 0)[0]:
            if i < 59: continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            e = i+1; x = i+20
            if e >= n or x >= n or o[e] <= 0: continue
            half = ROUND_TRIP/2
            for t in range(e, x+1):
                d = dates[t].replace("-", "")
                if d not in didx: continue
                r = (c[t]/c[t-1]-1)*PCT if t > e else (c[t]/o[e]-1)*PCT
                if t == e: r -= half
                if t == x: r -= half
                pos_ret.setdefault(d, []).append(r)
    daily = {d: st.mean(v) for d, v in pos_ret.items()}
    return nav_of(daily)

# ---------- 基准(等权日收益 + 成本拖累) ----------
def bench_nav(mode):
    drag = ROUND_TRIP/20.0
    cnt = {}
    tot = {}
    for code, ft in feats1.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        c = ft["c"].astype(np.float64); dates = ft["date"]
        for t in range(1, len(c)):
            d = dates[t].replace("-", "")
            if d not in didx: continue
            r = (c[t]/c[t-1]-1)*PCT - drag
            cnt[d] = cnt.get(d, 0)+1; tot[d] = tot.get(d, 0.0)+r
    daily = {d: tot[d]/cnt[d] for d in tot}
    return nav_of(daily)

out = {}
for name, fn in [("B2_全A", lambda: signal_nav(feats2, "b2_sig", "all")),
                 ("B2_强信号", lambda: signal_nav(feats2, "b2_strong", "all")),
                 ("B2_池内", lambda: signal_nav(feats2, "b2_sig", "pool")),
                 ("B1_全A", lambda: signal_nav(feats1, "sig_base", "all")),
                 ("B1_池内", lambda: signal_nav(feats1, "sig_base", "pool")),
                 ("基准_等权全A", lambda: bench_nav("all")),
                 ("基准_等权池内", lambda: bench_nav("pool"))]:
    eq, m = fn()
    out[name] = {"metrics": m, "equity": {d: eq[d] for d in all_dates if d in eq}}
    a = m["全部"]; tr = m["训练期"]; va = m["验证期"]
    print("%-12s 累计%+8.1f%% 年化%+7.1f%% 波动%5.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练%+7.1f%% 验证%+7.1f%%" % (
        name, a["total"], a["ann"], a["vol"], a["sharpe"], a["maxdd"], tr["total"], va["total"]))

json.dump({k: {"metrics": v["metrics"]} for k, v in out.items()},
          open(os.path.join(HERE, "b2_nav_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump({k: v["equity"] for k, v in out.items()},
          open(os.path.join(HERE, "b2_nav_equity.json"), "w", encoding="utf-8"))
print("saved -> b2_nav_results.json / b2_nav_equity.json")
