# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step3 组合层绩效(与B1/B2同口径同基准)
- 信号模拟: T+1开盘入场, 20日, 止损档 [不止损/8/12/15/20%], 移动止盈(峰值50%回撤, 同原引擎)
- 费率: 单次往返0.30%(佣金0.025%x2 + 印花0.05% + 滑点0.1%x2)
- 计费方式: 月度等权组合, 每笔信号pnl-0.30%
- 基准: 等权全A / 等权池内(同b1管道口径)
- bp1特有一项: MAE分布 vs 止损档(止损与波动率失配分析, B1报告核心发现的延续)
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
STOPS = [None, 8, 12, 15, 20]
PCT = 100.0
COST = {"commission": 0.00025, "stamp": 0.0005, "slip": 0.001}
ROUND_TRIP = (COST["commission"]*2 + COST["stamp"] + COST["slip"]*2)*PCT

d = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def simulate(arr, sd, stop_pct, hold=20):
    """T+1开盘入场, hold日, 止损stop_pct%(日內低点触发), 移动止盈=自峰值收盘回撤50%"""
    idx = None
    for i, row in enumerate(arr):
        if row[0] > sd: idx = i; break
    if idx is None: return None, None
    entry = arr[idx][1]
    if not entry or entry <= 0: return None, None
    sp = entry*(1-stop_pct/100.0) if stop_pct else None
    peak = entry; exit_p = None; reason = None
    last = min(idx+hold-1, len(arr)-1)
    for j in range(idx, last+1):
        _, o, h, l, c = arr[j]
        if sp is not None and l is not None and l <= sp:
            exit_p = sp; reason = "止损"; break
        if c:
            if c > peak: peak = c
            if j > idx and peak and c <= peak*0.5:
                exit_p = c; reason = "移动止盈"; break
        if j == last: exit_p = c if c else exit_p; reason = reason or "到期"
    if exit_p is None or not exit_p: return None, None
    return (exit_p/entry-1)*PCT, reason

kl = {code: [(ft["date"][i].replace("-", ""), float(ft["o"][i]), float(ft["h"][i]), float(ft["l"][i]), float(ft["c"][i]))
             for i in range(len(ft["date"]))] for code, ft in feats.items()}

def port_metrics(monthly):
    yms = sorted(monthly); rs = [monthly[y] for y in yms]; n = len(rs)
    if n < 2 or st.pstdev(rs) == 0: return None
    eq, cur = [], 1.0
    for r in rs: cur *= (1+r/100.0); eq.append(cur)
    mu, sd_ = st.mean(rs), st.pstdev(rs)
    peak, mdd = eq[0], 0.0
    for v in eq:
        if v > peak: peak = v
        dd = v/peak-1
        if dd < mdd: mdd = dd
    return {"n_months": n, "total": (cur-1)*100, "ann": (cur**(12.0/n)-1)*100,
            "sharpe": mu/sd_*math.sqrt(12), "maxdd": mdd*100, "mean_m": mu,
            "win_m": 100*sum(1 for r in rs if r > 0)/n}

sigs = []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["bp1_sig"])[0]:
        ds = dates[i].replace("-", "")
        if D0 <= ds <= D1: sigs.append((code, ds, 1 if code in pool_codes else 0))
print("模拟信号: %d (池内 %d)" % (len(sigs), sum(s[2] for s in sigs)))

results = {}
for sp in STOPS:
    label = "不止损" if sp is None else "止损%d%%" % sp
    monthly = {}; reasons = {}; per_sig = []; stop_cost = []
    for code, ds, ip in sigs:
        pnl, reason = simulate(kl[code], ds, sp)
        if pnl is None: continue
        monthly.setdefault(ds[:6], []).append(pnl)
        reasons[reason] = reasons.get(reason, 0)+1
        per_sig.append((pnl, period_of(ds)))
        if reason == "止损":
            base, _ = simulate(kl[code], ds, None)
            if base is not None: stop_cost.append(base)
    per = {}
    for pname in ["训练期", "验证期"]:
        pl = [x[0] for x in per_sig if x[1] == pname]
        if pl: per[pname] = {"n": len(pl), "mean": st.mean(pl), "median": st.median(pl),
                             "win": 100*sum(1 for v in pl if v > 0)/len(pl)}
    agg = {}
    for pname in ["训练期", "验证期"]:
        m = {ym: st.mean(v) for ym, v in monthly.items() if period_of(ym+"01") == pname}
        pm = port_metrics(m)
        if pm: agg[pname] = pm
    a = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    results[label] = {"portfolio_all": a, "portfolio_period": agg, "signal_level": per,
                      "exit_reasons": reasons, "n": len(per_sig),
                      "stop_opp_cost": {"n": len(stop_cost), "mean_if_no_stop": st.mean(stop_cost) if stop_cost else None}}
    print("%-8s n=%6d 累计%+8.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练%+6.2f%%/%.1f%% 验证%+6.2f%%/%.1f%% | %s" % (
        label, len(per_sig), a["total"], a["sharpe"], a["maxdd"],
        per["训练期"]["mean"], per["训练期"]["win"], per["验证期"]["mean"], per["验证期"]["win"], reasons))
    if stop_cost:
        print("         止损代价: 被止损的%d笔若不止损 mean=%+.2f%%" % (len(stop_cost), st.mean(stop_cost)))

# ---------- 基准(等权全A/池内, 从bp1管道全股票重算) ----------
def baseline_portfolio(mode):
    monthly = {}
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        t1 = ft["t1_20"]; dates = ft["date"]
        for i in range(59, len(t1)-20):
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if D0 <= ds <= D1: monthly.setdefault(ds[:6], []).append(float(t1[i])*PCT)
    out = {}
    for pname in ["训练期", "验证期", "全部"]:
        m = ({ym: st.mean(v) for ym, v in monthly.items()} if pname == "全部"
             else {ym: st.mean(v) for ym, v in monthly.items() if period_of(ym+"01") == pname})
        pm = port_metrics(m)
        if pm: out[pname] = pm
    return out

bench_port = {"等权池内": baseline_portfolio("pool"), "等权全A": baseline_portfolio("all"), "等权池外": baseline_portfolio("out")}
for k, v in bench_port.items():
    a = v["全部"]; print("%-8s 累计%+8.1f%% 夏普%+5.2f 回撤%6.1f%%" % (k, a["total"], a["sharpe"], a["maxdd"]))

# ---------- 计费 ----------
costed = {}
for label, r in results.items():
    sp = None if label == "不止损" else int(label.replace("止损", "").replace("%", ""))
    monthly = {}
    for code, ds, ip in sigs:
        pnl, _ = simulate(kl[code], ds, sp)
        if pnl is not None: monthly.setdefault(ds[:6], []).append(pnl-ROUND_TRIP)
    a = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    costed[label] = a
    print("计费 %-8s 累计%+8.1f%% 夏普%+5.2f" % (label, a["total"], a["sharpe"]))

# ---------- 池内组合 ----------
pool_out = {}
for sp in STOPS:
    label = "不止损" if sp is None else "止损%d%%" % sp
    monthly = {}
    for code, ds, ip in sigs:
        if not ip: continue
        pnl, _ = simulate(kl[code], ds, sp)
        if pnl is not None: monthly.setdefault(ds[:6], []).append(pnl)
    a = port_metrics({ym: st.mean(v) for ym, v in monthly.items()})
    per = {}
    for pname in ["训练期", "验证期"]:
        m = {ym: st.mean(v) for ym, v in monthly.items() if period_of(ym+"01") == pname}
        pm = port_metrics(m)
        if pm: per[pname] = pm
    monthly_c = {ym: st.mean([v-ROUND_TRIP for v in vs]) for ym, vs in monthly.items()}
    ac = port_metrics(monthly_c)
    pool_out[label] = {"all": a, "period": per, "costed": ac}
    print("池内 %-8s 累计%+7.1f%% 夏普%+5.2f 回撤%6.1f%% | 训练%+7.1f%% 验证%+7.1f%% | 计费%+7.1f%%" % (
        label, a["total"], a["sharpe"], a["maxdd"], per["训练期"]["total"], per["验证期"]["total"], ac["total"]))

# ---------- MAE 与止损失配 ----------
mae_all = []
for code, ft in feats.items():
    m = ft["mae20"]; dates = ft["date"]
    for i in np.where(ft["bp1_sig"])[0]:
        if np.isnan(m[i]): continue
        ds = dates[i].replace("-", "")
        if D0 <= ds <= D1: mae_all.append(float(m[i])*PCT)
mae_all = np.array(mae_all)
mismatch = {"n": len(mae_all), "median": float(np.median(mae_all)), "mean": float(mae_all.mean()),
            "frac_below": {str(s): float(100*(mae_all <= -s).mean()) for s in [5, 8, 10, 12, 15, 20, 25]}}
print("\nMAE(信号期最大不利偏移): 中位%.2f%% 均值%.2f%% | %s" % (mismatch["median"], mismatch["mean"], mismatch["frac_below"]))

json.dump({"results": results, "bench_port": bench_port, "costed": costed, "pool": pool_out,
           "mae_mismatch": mismatch, "cost_model": COST, "round_trip_pct": ROUND_TRIP},
          open(os.path.join(HERE, "bp1_port_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> bp1_port_results.json")
