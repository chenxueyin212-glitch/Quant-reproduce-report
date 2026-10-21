# -*- coding: utf-8 -*-
"""
B1因子重构评估 — Step3 组合层绩效
口径(与原引擎 metrics_all_signals.py 校准一致):
  T日信号 → T+1开盘入场 → T+20收盘出场(20个交易日)
  止损档: 无 / 8% / 12% / 15% / 20% (入场价下方, 最低价触及即成交, 缺口不特殊处理)
  移动止盈: 峰值收盘价回撤50% (原引擎语义)
  组合: 月度cohort等权(当月信号平均pnl=该cohort月收益), 不计成本为基准口径, 另报计费口径
基准: 等权池内 / 等权全A / 中证500(000905) / 上证(000001, 仅2025-09后)
输出: b1_port_results.json
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
STOPS = [None, 8, 12, 15, 20]
PCT = 100.0
COST = {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001}   # 双边/卖出/双边, 单次往返≈0.275%
ROUND_TRIP = COST["commission"]*2 + COST["stamp"] + COST["slip"]*2

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats, bench = d["feats"], d["bench"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def simulate(arr, sd, stop_pct, hold=20):
    """arr: [(date,o,h,l,c),...] 原引擎 run_one 语义"""
    idx = None
    for i, row in enumerate(arr):
        if row[0] > sd: idx = i; break
    if idx is None: return None, None
    entry = arr[idx][1]
    if not entry or entry <= 0: return None, None
    stop_price = entry*(1-stop_pct/100.0) if stop_pct else None
    peak = entry; exit_p = None; reason = None
    last = min(idx+hold-1, len(arr)-1)
    for j in range(idx, last+1):
        _, o, h, l, c = arr[j]
        if stop_price is not None and l is not None and l <= stop_price:
            exit_p = stop_price; reason = "止损"; break
        if c:
            if c > peak: peak = c
            if j > idx and peak and c <= peak*0.5:
                exit_p = c; reason = "移动止盈"; break
        if j == last:
            exit_p = c if c else exit_p; reason = reason or "到期"
    if exit_p is None or not exit_p: return None, None
    return (exit_p/entry-1)*PCT, reason

# ---------- 预取每只股票K线为list ----------
kl = {}
for code, ft in feats.items():
    dates = ft["date"]; n = len(dates)
    kls = [(dates[i].replace("-", ""), float(ft["o"][i]), float(ft["h"][i]), float(ft["l"][i]), float(ft["c"][i]))
           for i in range(n)]
    kl[code] = kls

def port_metrics(monthly):
    yms = sorted(monthly); rs = [monthly[y] for y in yms]; n = len(rs)
    if n < 2 or st.pstdev(rs) == 0: return None
    eq, cur = [], 1.0
    for r in rs: cur *= (1+r/100.0); eq.append(cur)
    total = cur; ann = (total**(12.0/n)-1)*100
    mu, sd_ = st.mean(rs), st.pstdev(rs)
    sharpe = mu/sd_*math.sqrt(12)
    peak, mdd = eq[0], 0.0
    for v in eq:
        if v > peak: peak = v
        dd = v/peak-1
        if dd < mdd: mdd = dd
    return {"n_months": n, "total": (total-1)*100, "ann": ann, "sharpe": sharpe,
            "maxdd": mdd*100, "mean_m": mu, "win_m": 100*sum(1 for r in rs if r > 0)/n}

# ---------- 信号模拟 ----------
sigs = []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        sigs.append((code, ds, "b1_dc" if ft["sig_dc"][i] else "b1_dk", 1 if code in pool_codes else 0))
print("模拟信号: %d" % len(sigs))

results = {}
for sp in STOPS:
    label = "不止损" if sp is None else "止损%d%%" % sp
    monthly = {}
    reasons = {}
    per_sig = []   # (pnl, period, in_pool)
    stop_cost = [] # 被止损交易的不止损20日收益(机会成本)
    for code, ds, t, ip in sigs:
        arr = kl[code]
        pnl, reason = simulate(arr, ds, sp)
        if pnl is None: continue
        ym = ds[:6]
        monthly.setdefault(ym, []).append(pnl)
        reasons[reason] = reasons.get(reason, 0) + 1
        per_sig.append((pnl, period_of(ds), ip))
        if reason == "止损":
            base, _ = simulate(arr, ds, None)
            if base is not None: stop_cost.append(base)
    agg = {}
    per = {}
    for (pname), pl in [(p, [x[0] for x in per_sig if x[1] == p]) for p in ["训练期", "验证期"]]:
        if pl:
            per[pname] = {"n": len(pl), "mean": st.mean(pl), "median": st.median(pl),
                          "win": 100*sum(1 for v in pl if v > 0)/len(pl)}
    for pname in ["训练期", "验证期"]:
        m = {ym: st.mean(v) for ym, v in monthly.items() if period_of(ym+"01") == pname}
        pm = port_metrics(m)
        if pm: agg[pname] = pm
    m_all = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    results[label] = {"portfolio_all": m_all, "portfolio_period": agg, "signal_level": per,
                      "exit_reasons": reasons, "n": len(per_sig),
                      "stop_opp_cost": {"n": len(stop_cost), "mean_if_no_stop": st.mean(stop_cost) if stop_cost else None}}
    pm = m_all; sl = per
    print("%-8s n=%6d 月=%d 累计%+8.1f%% 年化%+7.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练 mean%+6.2f%%/win%.1f%% 验证 mean%+6.2f%%/win%.1f%% | 出场%s" % (
        label, len(per_sig), pm["n_months"], pm["total"], pm["ann"], pm["sharpe"], pm["maxdd"],
        per["训练期"]["mean"], per["训练期"]["win"], per["验证期"]["mean"], per["验证期"]["win"], reasons))

# ---------- 基准组合(同构造) ----------
def baseline_portfolio(mode):
    monthly = {}
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        dates = ft["date"]; t1 = ft["t1_20"]
        for i in range(59, len(dates)-20):
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            monthly.setdefault(ds[:6], []).append(float(t1[i])*PCT)
    out = {}
    for pname in ["训练期", "验证期", "全部"]:
        if pname == "全部":
            m = {ym: st.mean(v) for ym, v in monthly.items()}
        else:
            m = {ym: st.mean(v) for ym, v in monthly.items() if period_of(ym+"01") == pname}
        pm = port_metrics(m)
        if pm: out[pname] = pm
    return out

bench_port = {"等权池内": baseline_portfolio("pool"), "等权全A": baseline_portfolio("all"), "等权池外": baseline_portfolio("out")}
for k, v in bench_port.items():
    a = v.get("全部")
    if a: print("%-8s 月=%d 累计%+8.1f%% 年化%+7.1f%% 夏普%+5.2f 回撤%6.1f%%" % (k, a["n_months"], a["total"], a["ann"], a["sharpe"], a["maxdd"]))

# ---------- 指数基准(月度) ----------
def index_monthly(code):
    b = bench[code]; dates = b["date"]; cl = b["close"]
    mret = {}; prev = None
    for i, ds in enumerate(dates):
        ym = ds.replace("-", "")[:6]
        if prev is not None and ym != prev[0]:
            mret[ym] = (cl[i]/prev[1]-1)*PCT
        prev = (ym, cl[i])
    return mret

for ic in ["000905", "000001"]:
    im = index_monthly(ic)
    for pname in ["全部"]:
        if pname == "全部": m = im
        else: m = {k: v for k, v in im.items() if period_of(k+"01") == pname}
        pm = port_metrics(m)
        if pm: print("%s(%s) 月=%d 累计%+8.1f%% 年化%+7.1f%% 夏普%+5.2f 回撤%6.1f%%" % ("中证500" if ic == "000905" else "上证指数", pname, pm["n_months"], pm["total"], pm["ann"], pm["sharpe"], pm["maxdd"]))

# ---------- 计费口径(在无止损月度均值上扣往返成本) ----------
costed = {}
for label, r in results.items():
    sp = None if label == "不止损" else int(label.replace("止损", "").replace("%", ""))
    monthly = {}
    for code, ds, t, ip in sigs:
        arr = kl[code]
        pnl, reason = simulate(arr, ds, sp)
        if pnl is None: continue
        monthly.setdefault(ds[:6], []).append(pnl - ROUND_TRIP*PCT)
    a = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    costed[label] = a
    if a: print("计费 %-8s 累计%+8.1f%% 年化%+7.1f%% 夏普%+5.2f 回撤%6.1f%% (往返成本%.3f%%)" % (label, a["total"], a["ann"], a["sharpe"], a["maxdd"], ROUND_TRIP*PCT))

# ---------- 止损失配: mae20分布 vs 止损档 ----------
mae_all = []
for code, ft in feats.items():
    m = ft["mae20"]
    for i in range(59, len(m)):
        if np.isnan(m[i]): continue
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if ft["sig_base"][i]: mae_all.append(float(m[i])*PCT)
mae_all = np.array(mae_all)
mismatch = {"n": len(mae_all), "median": float(np.median(mae_all)),
            "mean": float(mae_all.mean()),
            "frac_below": {str(s): float(100*(mae_all <= -s).mean()) for s in [5, 8, 10, 12, 15, 20, 25]}}
print("\n信号20日最大不利偏移(MAE)分布: n=%d 中位%.2f%% 均值%.2f%%" % (mismatch["n"], mismatch["median"], mismatch["mean"]))
for s, f in mismatch["frac_below"].items(): print("  MAE≤−%s%%: %.1f%%" % (s, f))

json.dump({"results": {k: {"portfolio_all": v["portfolio_all"], "portfolio_period": v["portfolio_period"],
                           "signal_level": v["signal_level"], "exit_reasons": v["exit_reasons"],
                           "n": v["n"], "stop_opp_cost": v["stop_opp_cost"]} for k, v in results.items()},
           "bench_port": bench_port, "costed": costed, "mae_mismatch": mismatch,
           "cost_model": COST, "round_trip_pct": ROUND_TRIP*PCT},
          open(os.path.join(HERE, "b1_port_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> b1_port_results.json")
