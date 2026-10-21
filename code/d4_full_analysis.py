# -*- coding: utf-8 -*-
"""
d4 波动率压缩伏击因子 — S级产品化完整五项检验
信号: d4>=15 & platform_range<25 & ~bp1_tech（未突破）
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20)，训练<20250908，费率0.30%/往返
模块: 信号层 / 组合层(月度cohort+日度等权NAV) / 稳健性 / 微观
"""
import pickle, os, json, math, statistics as st
import numpy as np
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
ROUND_TRIP = (0.00025*2 + 0.0005 + 0.001*2)*PCT   # 0.30%

d = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

# 【数据清洗 D29】日期格式归一：特征文件混入 4 只股票 928 条 YYYY-MM-DD 连字符日期，
# 字符串排序会错位（"2026-01-05" < "20260101"），在 NAV 逐日复利中形成幽灵交易日。
# 统一压成 YYYYMMDD 再参与任何日期比较/排序/分桶。
for _ft in feats.values():
    _ft["date"] = np.array([str(x).replace("-", "") for x in _ft["date"]])

print("宇宙 %d 只, 池 %d 只" % (len(feats), len(pool_codes)))

all_dates = sorted({str(ds) for ft in feats.values() for ds in ft["date"]})
didx = {dd: i for i, dd in enumerate(all_dates)}

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std())}

R = {}

# ================= 信号构建 =================
def mask_d4(ft, thr=15, plat_max=25, exclude_tech=True):
    d4 = ft["d4"]; plat = ft["platform_range"]; tech = ft["bp1_tech"] > 0 if ft["bp1_tech"].dtype != bool else ft["bp1_tech"]
    m = (d4 >= thr) & (plat < plat_max) & ~np.isnan(d4) & ~np.isnan(plat)
    if exclude_tech: m = m & (~tech)
    return m

rows = []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]; o = ft["o"].astype(np.float64); c = ft["c"].astype(np.float64)
    m = mask_d4(ft)
    in_pool = code in pool_codes
    # fresh/sustained: 状态持续天数
    d4 = ft["d4"]
    state = ((d4 >= 15) & ~np.isnan(d4) & (ft["platform_range"] < 25)).astype(np.int8)
    # run length
    run = np.zeros(len(state), np.int32)
    for i in range(1, len(state)):
        run[i] = run[i-1]+1 if state[i] else 0
    for i in np.where(m)[0]:
        if np.isnan(t1[i]): continue
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        rows.append({"code": code, "date": ds, "in_pool": in_pool, "period": period_of(ds),
                     "t1_20": float(t1[i])*PCT,
                     "d4": float(d4[i]), "plat": float(ft["platform_range"][i]),
                     "vatr": float(ft["vol_atr_ratio"][i]) if not np.isnan(ft["vol_atr_ratio"][i]) else None,
                     "brk_amp": float(ft["breakout_amp"][i]) if not np.isnan(ft["breakout_amp"][i]) else None,
                     "run": int(run[i]),
                     "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
                     "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
                     "below_days": float(ft["below_days"][i]),
                     "first5_min": float(ft["first5_min"][i])*PCT,
                     "last15_max": float(ft["last15_max"][i])*PCT})
print("有效信号 %d 条" % len(rows))
R["meta"] = {"n_signals": len(rows), "n_raw": int(sum(mask_d4(ft).sum() for ft in feats.values()))}
R["truncation"] = {"n_total": R["meta"]["n_raw"], "n_truncated": R["meta"]["n_raw"] - len(rows)}

# ================= 1. 信号层 =================
groups = {}
groups["全体"] = desc([r["t1_20"] for r in rows])
groups["训练期"] = desc([r["t1_20"] for r in rows if r["period"] == "训练期"])
groups["验证期"] = desc([r["t1_20"] for r in rows if r["period"] == "验证期"])
groups["池内"] = desc([r["t1_20"] for r in rows if r["in_pool"]])
groups["池外"] = desc([r["t1_20"] for r in rows if not r["in_pool"]])
for p in ["训练期", "验证期"]:
    for ip, nm in [(True, "池内"), (False, "池外")]:
        groups["%s-%s" % (nm, p)] = desc([r["t1_20"] for r in rows if r["period"] == p and r["in_pool"] == ip])
groups["run=1(fresh)"] = desc([r["t1_20"] for r in rows if r["run"] == 1])
groups["run 2-5"] = desc([r["t1_20"] for r in rows if 2 <= r["run"] <= 5])
groups["run >5(sustained)"] = desc([r["t1_20"] for r in rows if r["run"] > 5])
groups["d4 15-18"] = desc([r["t1_20"] for r in rows if r["d4"] < 18])
groups["d4 >=18"] = desc([r["t1_20"] for r in rows if r["d4"] >= 18])
R["groups"] = groups

# deciles by t1_20
vals = sorted(rows, key=lambda r: r["t1_20"])
n = len(vals); dec = {}
for k in range(10):
    seg = vals[int(n*k/10):int(n*(k+1)/10)]
    if not seg: continue
    arr = [s["t1_20"] for s in seg]
    tot = sum(x["t1_20"] for x in vals)
    dec["D%d(%.1f~%.1f%%)" % (k+1, seg[0]["t1_20"], seg[-1]["t1_20"])] = {
        "n": len(seg), "mean": float(np.mean(arr)), "pnl_share": 100.0*sum(arr)/tot if tot else None}
R["deciles"] = dec

# IC：d4 是 0-15 整数评分且封顶 15，信号组内截面恒等于 15（std=0），
# 组内 IC 不可定义 → 【口径说明】改算两套：
#   (a) 全宇宙日截面 IC：当日全 A 每股 d4(0-15) vs t1_20 —— 因子排序力的主检验
#   (b) 信号组内 IC：如实登记为「退化/不可定义」
from scipy.stats import spearmanr
def ic_series(get_f, get_t, min_n=3):
    daily = defaultdict(list)
    for code, ft in feats.items():
        t1 = ft["t1_20"]; dates = ft["date"]
        farr = get_f(ft)
        for i in range(len(dates)):
            if np.isnan(t1[i]) or farr is None: continue
            fv = farr[i]
            if np.isnan(fv): continue
            ds = str(dates[i])
            if not (D0 <= ds <= D1): continue
            daily[ds].append((float(fv), float(t1[i])*PCT))
    ics = []
    for ds, lst in daily.items():
        if len(lst) < min_n: continue
        f = np.array([x[0] for x in lst]); t = np.array([x[1] for x in lst])
        if np.std(f) == 0: continue
        ic, _ = spearmanr(f, t)
        if not np.isnan(ic): ics.append(float(ic))
    arr = np.array(ics)
    if len(arr) == 0:
        return {"n_days": 0, "mean": None, "icir": None, "pos_ratio": None, "t": None, "note": "退化/不可定义"}
    return {"n_days": int(len(arr)), "mean": float(arr.mean()),
            "icir": float(arr.mean()/arr.std()) if arr.std() > 0 else None,
            "pos_ratio": float(100*(arr > 0).mean()),
            "t": float(arr.mean()/(arr.std()/np.sqrt(len(arr)))) if arr.std() > 0 else None,
            "note": None}

R["ic"] = {
    "universe_d4": ic_series(lambda ft: ft["d4"], None),
    "signal_group_d4": {"n_days": 0, "mean": None, "icir": None, "pos_ratio": None, "t": None,
                        "note": "退化/不可定义：d4>=15 组内 d4 恒等于 15（d4 封顶 15），截面 std=0"},
    "signal_group_plat": ic_series(lambda ft: np.where(mask_d4(ft), ft["platform_range"], np.nan), None),
}

# naive
base = {"all": [], "train": [], "valid": []}
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    for i in range(len(dates)):
        if np.isnan(t1[i]): continue
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        v = float(t1[i])*PCT
        base["all"].append(v)
        base["train" if ds < TRAIN_END else "valid"].append(v)
R["naive"] = {"伏击信号(d4>=15)": desc([r["t1_20"] for r in rows]),
              "全宇宙等权(同持有期)": desc(base["all"]),
              "全宇宙-训练": desc(base["train"]), "全宇宙-验证": desc(base["valid"])}

# ================= 2. 组合层 =================
def cohort(sim_rows, stop=None):
    mon = defaultdict(list)
    for r in sim_rows:
        pnl = r["t1_20"]
        if stop is not None and r["mae20"] <= -stop: pnl = -stop
        mon[r["date"][:6]].append(pnl)
    mrets = np.array([np.mean(mon[m]) for m in sorted(mon)])
    if len(mrets) == 0: return None
    nav = np.cumprod(1 + mrets/100.0)
    peak = np.maximum.accumulate(nav); dd = (nav/peak-1)*100
    return {"n_months": int(len(mrets)), "total": float((nav[-1]-1)*100),
            "ann": float((nav[-1]**(12.0/len(mrets))-1)*100),
            "sharpe": float(mrets.mean()/mrets.std()*np.sqrt(12)) if mrets.std() > 0 else None,
            "maxdd": float(dd.min()), "win_m": float(100*(mrets > 0).mean()),
            "monthly": {m: float(v) for m, v in zip(sorted(mon), mrets)}}

port = {}
for nm, stop in [("不止损", None), ("止损8%", 8), ("止损12%", 12), ("止损15%", 15)]:
    port[nm] = {"all": cohort(rows, stop),
                "train": cohort([r for r in rows if r["period"] == "训练期"], stop),
                "valid": cohort([r for r in rows if r["period"] == "验证期"], stop)}
R["port"] = port
R["pool_port"] = {"不止损": {"all": cohort([r for r in rows if r["in_pool"]]),
                             "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"]),
                             "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"])},
                  "止损8%": {"all": cohort([r for r in rows if r["in_pool"]], 8),
                             "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"], 8),
                             "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"], 8)}}

# 基准 cohort
bench_all, bench_pool = [], []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    for i in range(len(dates)):
        if np.isnan(t1[i]): continue
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        rec = {"date": ds, "t1_20": float(t1[i])*PCT, "mae20": -999.0,
               "period": period_of(ds), "in_pool": code in pool_codes}
        bench_all.append(rec)
        if rec["in_pool"]: bench_pool.append(rec)
R["bench_port"] = {"等权全A": {"all": cohort(bench_all),
                                "train": cohort([r for r in bench_all if r["period"] == "训练期"]),
                                "valid": cohort([r for r in bench_all if r["period"] == "验证期"])},
                    "等权池内": {"all": cohort(bench_pool),
                                "train": cohort([r for r in bench_pool if r["period"] == "训练期"]),
                                "valid": cohort([r for r in bench_pool if r["period"] == "验证期"])}}

# ---- 日度等权再平衡 NAV (同 bp1 口径) ----
def nav_of(daily_rets):
    eq, cur = {}, 1.0
    for dd in all_dates:
        cur *= (1 + daily_rets.get(dd, 0.0)/100.0); eq[dd] = cur
    def metrics(lo, hi):
        seg = [eq[dd] for dd in all_dates if lo <= dd < hi]
        if len(seg) < 20: return None
        rs = np.diff(np.log(seg))
        ann = (seg[-1]/seg[0])**(244.0/len(seg))-1
        vol = float(rs.std()*math.sqrt(244))
        peak = 0.0; mdd = 0.0
        for v in seg:
            if v > peak: peak = v
            dd_ = v/peak-1
            if dd_ < mdd: mdd = dd_
        return {"total": (seg[-1]/seg[0]-1)*100, "ann": ann*100, "vol": vol*100,
                "sharpe": float(rs.mean()/rs.std()*math.sqrt(244)) if rs.std() > 0 else None,
                "maxdd": mdd*100}
    return eq, {"全部": metrics(D0, D1+"z"), "训练期": metrics(D0, TRAIN_END), "验证期": metrics(TRAIN_END, D1+"z")}

def signal_nav(mode="all"):
    pos_ret = {}
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64); dates = ft["date"]
        n = len(c)
        m = mask_d4(ft)
        for i in np.where(m)[0]:
            if i < 59: continue
            ds = str(dates[i])
            if not (D0 <= ds <= D1): continue
            e = i+1; x = i+20
            if e >= n or x >= n or o[e] <= 0: continue
            half = ROUND_TRIP/2
            for t in range(e, x+1):
                dd = str(dates[t])
                if dd not in didx: continue
                r = (c[t]/c[t-1]-1)*PCT if t > e else (c[t]/o[e]-1)*PCT
                if t == e: r -= half
                if t == x: r -= half
                pos_ret.setdefault(dd, []).append(r)
    daily = {dd: st.mean(v) for dd, v in pos_ret.items()}
    return nav_of(daily)

def bench_nav(mode):
    drag = ROUND_TRIP/20.0
    tot = {}; cnt = {}
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        c = ft["c"].astype(np.float64); dates = ft["date"]
        for t in range(1, len(c)):
            dd = str(dates[t])
            if dd not in didx or c[t-1] <= 0: continue
            tot[dd] = tot.get(dd, 0.0) + (c[t]/c[t-1]-1)*PCT - drag
            cnt[dd] = cnt.get(dd, 0) + 1
    daily = {dd: tot[dd]/cnt[dd] for dd in tot if cnt[dd]}
    return nav_of(daily)

R["nav"] = {}
EQ = {}
for nm, fn in [("d4_全A", lambda: signal_nav("all")), ("d4_池内", lambda: signal_nav("pool")),
               ("d4_池外", lambda: signal_nav("out")),
               ("基准_等权全A", lambda: bench_nav("all")), ("基准_等权池内", lambda: bench_nav("pool")),
               ("基准_等权池外", lambda: bench_nav("out"))]:
    eq, met = fn()
    R["nav"][nm] = {"metrics": met}
    EQ[nm] = eq
    print("NAV %-12s total=%+.1f%% sharpe=%+.2f" % (nm, met["全部"]["total"], met["全部"]["sharpe"]))
json.dump(EQ, open(os.path.join(HERE, "d4_nav_equity.json"), "w"))
print("saved -> d4_nav_equity.json")

# MAE 失配
maes = np.array([r["mae20"] for r in rows])
R["mae_mismatch"] = {"n": int(len(maes)), "median": float(np.median(maes)), "mean": float(maes.mean()),
                     "frac_below": {str(t): float(100*(maes <= -t).mean()) for t in [5, 8, 10, 12, 15, 20]}}

# ================= 3. 稳健性 =================
ofat = {}
ofat["基准(d4>=15,plat<25)"] = desc([r["t1_20"] for r in rows])
for thr, lbl in [(12, "d4>=12"), (18, "d4>=18"), (20, "d4>=20")]:
    sub = []
    for code, ft in feats.items():
        t1 = ft["t1_20"]; dates = ft["date"]
        m = mask_d4(ft, thr=thr)
        for i in np.where(m)[0]:
            if np.isnan(t1[i]): continue
            ds = str(dates[i])
            if D0 <= ds <= D1: sub.append(float(t1[i])*PCT)
    ofat[lbl] = desc(sub)
for pm, lbl in [(20, "plat<20%"), (30, "plat<30%")]:
    sub = []
    for code, ft in feats.items():
        t1 = ft["t1_20"]; dates = ft["date"]
        m = mask_d4(ft, thr=15, plat_max=pm)
        for i in np.where(m)[0]:
            if np.isnan(t1[i]): continue
            ds = str(dates[i])
            if D0 <= ds <= D1: sub.append(float(t1[i])*PCT)
    ofat[lbl] = desc(sub)
# 持有期 10/30
for hold, lbl in [(10, "持有10日"), (30, "持有30日")]:
    sub = []
    for code, ft in feats.items():
        dates = ft["date"]; o = ft["o"].astype(np.float64); c = ft["c"].astype(np.float64)
        n = len(c)
        m = mask_d4(ft)
        for i in np.where(m)[0]:
            e = i+1; x = i+hold
            if e >= n or x >= n or o[e] <= 0: continue
            ds = str(dates[i])
            if not (D0 <= ds <= D1): continue
            sub.append((c[x]/o[e]-1)*PCT)
    ofat[lbl] = desc(sub)
ofat["含突破(不排除tech)"] = desc(
    [float(ft["t1_20"][i])*PCT for code, ft in feats.items() for i in np.where(mask_d4(ft, exclude_tech=False))[0]
     if not np.isnan(ft["t1_20"][i]) and D0 <= str(ft["date"][i]) <= D1])
R["ofat"] = ofat

R["regime"] = {
    "pre20>0": desc([r["t1_20"] for r in rows if r["pre20"] > 0]),
    "pre20<=0": desc([r["t1_20"] for r in rows if r["pre20"] <= 0]),
    "vol20<3%": desc([r["t1_20"] for r in rows if r["vol20"] < 3]),
    "vol20>=3%": desc([r["t1_20"] for r in rows if r["vol20"] >= 3]),
}

# ================= 4. 微观 =================
def layer(vals, keyfn):
    g = defaultdict(list)
    for r in vals:
        k = keyfn(r)
        if k: g[k].append(r["t1_20"])
    return {k: desc(v) for k, v in g.items()}

R["counterparty"] = {
    "pre20": layer(rows, lambda r: ("<0%" if r["pre20"] < 0 else "0-10%" if r["pre20"] < 10 else "10-20%" if r["pre20"] < 20 else "20-40%" if r["pre20"] < 40 else ">=40%")),
    "pre60": layer(rows, lambda r: ("<0%" if r["pre60"] < 0 else "0-20%" if r["pre60"] < 20 else "20-50%" if r["pre60"] < 50 else "50-100%" if r["pre60"] < 100 else ">=100%")),
    "dd20": layer(rows, lambda r: ("<-15%" if r["dd20"] < -15 else "-15~-10%" if r["dd20"] < -10 else "-10~-5%" if r["dd20"] < -5 else ">=-5%")),
    "vol20": layer(rows, lambda r: ("<2%" if r["vol20"] < 2 else "2-3%" if r["vol20"] < 3 else "3-4%" if r["vol20"] < 4 else ">=4%")),
}
R["quality"] = {
    "d4": layer(rows, lambda r: ("15-16" if r["d4"] < 16 else "16-18" if r["d4"] < 18 else ">=18")),
    "plat": layer(rows, lambda r: ("<15%" if r["plat"] < 15 else "15-20%" if r["plat"] < 20 else ">=20%")),
    "vatr": layer(rows, lambda r: None if r["vatr"] is None else ("<3%" if r["vatr"] < 3 else "3-5%" if r["vatr"] < 5 else ">=5%")),
    "run": layer(rows, lambda r: ("1(fresh)" if r["run"] == 1 else "2-5" if r["run"] <= 5 else ">5")),
}
mae20 = np.array([r["mae20"] for r in rows]); mfe20 = np.array([r["mfe20"] for r in rows])
f5 = np.array([r["first5_min"] for r in rows]); l15 = np.array([r["last15_max"] for r in rows])
bd = np.array([r["below_days"] for r in rows])
R["path"] = {"mae20": {"median": float(np.median(mae20)), "mean": float(mae20.mean())},
             "mfe20": {"median": float(np.median(mfe20)), "mean": float(mfe20.mean())},
             "first5_min": {"median": float(np.median(f5)), "mean": float(f5.mean())},
             "last15_max": {"median": float(np.median(l15)), "mean": float(l15.mean())},
             "below_days": {"median": float(np.median(bd)), "mean": float(bd.mean())}}

mon = defaultdict(list)
for r in rows: mon[r["date"][:6]].append(r["t1_20"])
R["monthly"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(mon.items())}

json.dump(R, open(os.path.join(HERE, "d4_full_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> d4_full_results.json")

# 摘要
print("\n=== 信号层 ===")
for k in ["全体","训练期","验证期","池内","池外","run=1(fresh)","run >5(sustained)","d4 15-18","d4 >=18"]:
    if R["groups"].get(k):
        v = R["groups"][k]
        print("%-22s n=%6d mean=%+.2f%% med=%+.2f%% win=%.1f%%" % (k, v["n"], v["mean"], v["median"], v["win"]))
icu = R["ic"]["universe_d4"]
if icu["mean"] is not None:
    print("IC 全宇宙d4: mean=%+.4f t=%+.2f icir=%.3f pos=%.1f%% (n_days=%d)" % (
        icu["mean"], icu["t"], icu["icir"], icu["pos_ratio"], icu["n_days"]))
print("IC 信号组内d4:", R["ic"]["signal_group_d4"]["note"])
icp = R["ic"]["signal_group_plat"]
if icp["mean"] is not None:
    print("IC 信号组内plat: mean=%+.4f t=%+.2f (n_days=%d)" % (icp["mean"], icp["t"], icp["n_days"]))
print("\n=== 组合 ===")
for k, v in R["port"].items():
    a = v["all"]
    if a: print("%s: total=%+.1f%% sharpe=%+.2f maxdd=%.1f%% tr=%+.1f%% va=%+.1f%%" % (
        k, a["total"], a["sharpe"], a["maxdd"], v["train"]["total"] if v["train"] else 0, v["valid"]["total"] if v["valid"] else 0))
print("NAV: d4全A=%+.1f%% vs 基准=%+.1f%% | d4池内=%+.1f%% vs 池内基准=%+.1f%%" % (
    R["nav"]["d4_全A"]["metrics"]["全部"]["total"], R["nav"]["基准_等权全A"]["metrics"]["全部"]["total"],
    R["nav"]["d4_池内"]["metrics"]["全部"]["total"], R["nav"]["基准_等权池内"]["metrics"]["全部"]["total"]))
print("\n=== OFAT ===")
for k, v in R["ofat"].items():
    if v: print("%-22s n=%6d mean=%+.2f%%" % (k, v["n"], v["mean"]))
print("MAE中位=%.2f%%" % R["mae_mismatch"]["median"])
