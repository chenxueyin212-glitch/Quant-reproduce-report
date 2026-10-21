# -*- coding: utf-8 -*-
"""
搬砖（砖型图VAR6A·多空临界）五项检验 — 信号层/组合层/NAV/稳健性/微观
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20)，训练<20250908，费率0.30%/往返
"""
import pickle, os, json, math, statistics as st
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
ROUND_TRIP = (0.00025*2 + 0.0005 + 0.001*2)*PCT   # 0.30%

d = pickle.load(open(os.path.join(HERE, "banzhuan_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = d["pool_codes"]
# 原始 OHLCV（NAV 逐日收益用）
_raw = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))["feats"]
# 【D29】日期归一（banzhuan_features 已归一，二次保险）
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

def layer(vals, split_by):
    g = defaultdict(list)
    for r in vals: g[split_by(r)].append(r["t1_20"])
    return {k: desc(v) for k, v in g.items()}

R = {}

# ================= 信号表 =================
rows = []
for code, ft in feats.items():
    dates = ft["date"]; sig = ft["sig"]; t1 = ft["t1_20"]
    in_pool = code in pool_codes
    for i in np.where(sig)[0]:
        if np.isnan(t1[i]): continue
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        rows.append({"code": code, "date": ds, "in_pool": in_pool,
                     "period": period_of(ds),
                     "t1_20": float(t1[i])*PCT,
                     "score": int(ft["score"][i]), "consec": int(ft["consec"][i]),
                     "var6a": float(ft["var6a"][i]),
                     "dist_low": float(ft["dist_low"][i]), "dist_close": float(ft["dist_close"][i]),
                     "bullish": bool(ft["bullish"][i]), "above_ma60": bool(ft["above_ma60"][i]),
                     "chg20d": float(ft["chg20d"][i]),
                     "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
                     "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
                     "below_days": float(ft["below_days"][i]),
                     "first5_min": float(ft["first5_min"][i])*PCT,
                     "last15_max": float(ft["last15_max"][i])*PCT})
print("有效信号 %d 条" % len(rows))

n_raw = sum(int(ft["sig"].sum()) for ft in feats.values())
R["truncation"] = {"n_total": n_raw, "n_truncated": n_raw - len(rows)}

# ---------- 1. 分组 ----------
groups = {}
groups["全体"] = desc([r["t1_20"] for r in rows])
groups["训练期"] = desc([r["t1_20"] for r in rows if r["period"] == "训练期"])
groups["验证期"] = desc([r["t1_20"] for r in rows if r["period"] == "验证期"])
groups["池内"] = desc([r["t1_20"] for r in rows if r["in_pool"]])
groups["池外"] = desc([r["t1_20"] for r in rows if not r["in_pool"]])
for p in ["训练期", "验证期"]:
    for ip, nm in [(True, "池内"), (False, "池外")]:
        groups["%s-%s" % (nm, p)] = desc([r["t1_20"] for r in rows if r["period"] == p and r["in_pool"] == ip])
# 砖位分层（机制主轴）
groups["第1砖"] = desc([r["t1_20"] for r in rows if r["consec"] == 1])
groups["第2砖"] = desc([r["t1_20"] for r in rows if r["consec"] == 2])
# score 分层
groups["score=5-6"] = desc([r["t1_20"] for r in rows if r["score"] <= 6])
groups["score=7"] = desc([r["t1_20"] for r in rows if r["score"] == 7])
groups["score>=8"] = desc([r["t1_20"] for r in rows if r["score"] >= 8])
# 阳线/MA60
groups["阳线(C>O)"] = desc([r["t1_20"] for r in rows if r["bullish"]])
groups["阴线"] = desc([r["t1_20"] for r in rows if not r["bullish"]])
groups["站上MA60"] = desc([r["t1_20"] for r in rows if r["above_ma60"]])
R["groups"] = groups

# ---------- 2. 十分位 ----------
vals = sorted(rows, key=lambda r: r["t1_20"])
n = len(vals); dec = {}
for k in range(10):
    seg = vals[int(n*k/10):int(n*(k+1)/10)]
    if not seg: continue
    arr = [s["t1_20"] for s in seg]
    tot = sum(x["t1_20"] for x in vals)
    dec["D%d(%.1f~%.1f%%)" % (k+1, seg[0]["t1_20"], seg[-1]["t1_20"])] = {
        "n": len(seg), "share": 100.0*len(seg)/n,
        "mean": float(np.mean(arr)), "pnl_share": 100.0*sum(arr)/tot if tot != 0 else None}
R["deciles"] = dec

# ---------- 3. 日度截面 IC ----------
from scipy.stats import spearmanr
daily = defaultdict(list)
for r in rows: daily[r["date"]].append(r)
R["ic"] = {}
for fac in ["score", "dist_low", "dist_close", "var6a", "chg20d"]:
    ics = []
    for ds, lst in daily.items():
        if len(lst) < 3: continue
        t = np.array([x["t1_20"] for x in lst])
        f = np.array([x[fac] for x in lst], dtype=float)
        if np.nanstd(f) == 0: continue
        ic, _ = spearmanr(f, t)
        if not np.isnan(ic): ics.append(ic)
    arr = np.array(ics)
    if len(arr) == 0: continue
    R["ic"][fac] = {"n_days": int(len(arr)), "mean": float(arr.mean()),
                    "icir": float(arr.mean()/arr.std()) if arr.std() > 0 else None,
                    "pos_ratio": float(100*(arr > 0).mean()),
                    "t": float(arr.mean()/(arr.std()/np.sqrt(len(arr)))) if arr.std() > 0 else None}

# ---------- 4. 朴素对照 ----------
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
R["naive"] = {
    "搬砖信号": desc([r["t1_20"] for r in rows]),
    "全宇宙等权(同持有期)": desc(base["all"]),
    "全宇宙-训练": desc(base["train"]),
    "全宇宙-验证": desc(base["valid"]),
}

# ---------- 5. 组合层(月度cohort) ----------
def cohort(sim_rows, stop=None):
    mon = defaultdict(list)
    for r in sim_rows:
        pnl = r["t1_20"]
        if stop is not None and r["mae20"] <= -stop:
            pnl = -stop
        mon[r["date"][:6]].append(pnl)
    mrets = [np.mean(mon[m]) for m in sorted(mon)]
    mrets = np.array(mrets)
    if len(mrets) == 0: return None
    nav = np.cumprod(1 + mrets/100.0)
    total = float((nav[-1]-1)*100)
    ann = float(((nav[-1])**(12.0/len(mrets))-1)*100)
    sharpe = float(mrets.mean()/mrets.std()*np.sqrt(12)) if mrets.std() > 0 else None
    peak = np.maximum.accumulate(nav); dd = (nav/peak-1)*100
    return {"n_months": int(len(mrets)), "total": total, "ann": ann, "sharpe": sharpe,
            "maxdd": float(dd.min()), "win_m": float(100*(mrets > 0).mean()),
            "monthly": {m: float(v) for m, v in zip(sorted(mon), mrets)}}

port = {}
for stop_name, stop in [("不止损", None), ("止损8%", 8), ("止损12%", 12), ("止损15%", 15)]:
    port[stop_name] = {
        "all": cohort(rows, stop),
        "train": cohort([r for r in rows if r["period"] == "训练期"], stop),
        "valid": cohort([r for r in rows if r["period"] == "验证期"], stop),
    }
R["port"] = port

R["pool_port"] = {
    "不止损": {"all": cohort([r for r in rows if r["in_pool"]]),
               "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"]),
               "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"])},
    "止损8%": {"all": cohort([r for r in rows if r["in_pool"]], 8),
               "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"], 8),
               "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"], 8)},
}

bench_rows_all = []; bench_rows_pool = []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    for i in range(len(dates)):
        if np.isnan(t1[i]): continue
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        rec = {"date": ds, "t1_20": float(t1[i])*PCT, "mae20": -999.0,
               "period": period_of(ds), "in_pool": code in pool_codes}
        bench_rows_all.append(rec)
        if rec["in_pool"]: bench_rows_pool.append(rec)
R["bench_port"] = {"等权全A": {"all": cohort(bench_rows_all),
                                "train": cohort([r for r in bench_rows_all if r["period"] == "训练期"]),
                                "valid": cohort([r for r in bench_rows_all if r["period"] == "验证期"])},
                    "等权池内": {"all": cohort(bench_rows_pool),
                                "train": cohort([r for r in bench_rows_pool if r["period"] == "训练期"]),
                                "valid": cohort([r for r in bench_rows_pool if r["period"] == "验证期"])}}

# ---------- 6. MAE 失配 ----------
maes = np.array([r["mae20"] for r in rows])
R["mae_mismatch"] = {
    "n": int(len(maes)),
    "median": float(np.median(maes)), "mean": float(maes.mean()),
    "frac_below": {str(t): float(100*(maes <= -t).mean()) for t in [5, 8, 10, 12, 15, 20]},
}

# ---------- 7. 日度等权再平衡 NAV（d4 口径，含费率） ----------
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
        rw = _raw[code]
        c = rw["c"].astype(np.float64); o = rw["o"].astype(np.float64); dates = ft["date"]
        n = len(c)
        for i in np.where(ft["sig"])[0]:
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
        c = _raw[code]["c"].astype(np.float64); dates = ft["date"]
        for t in range(1, len(c)):
            dd = str(dates[t])
            if dd not in didx or c[t-1] <= 0: continue
            tot[dd] = tot.get(dd, 0.0) + (c[t]/c[t-1]-1)*PCT - drag
            cnt[dd] = cnt.get(dd, 0) + 1
    daily = {dd: tot[dd]/cnt[dd] for dd in tot if cnt[dd]}
    return nav_of(daily)

R["nav"] = {}
EQ = {}
for nm, fn in [("bz_全A", lambda: signal_nav("all")), ("bz_池内", lambda: signal_nav("pool")),
               ("bz_池外", lambda: signal_nav("out")),
               ("基准_等权全A", lambda: bench_nav("all")), ("基准_等权池内", lambda: bench_nav("pool")),
               ("基准_等权池外", lambda: bench_nav("out"))]:
    eq, met = fn()
    R["nav"][nm] = {"metrics": met}
    EQ[nm] = eq
    m = met["全部"]
    if m: print("NAV %-12s total=%+.1f%% sharpe=%+.2f" % (nm, m["total"], m["sharpe"]))
json.dump(EQ, open(os.path.join(HERE, "banzhuan_nav_equity.json"), "w"))

# 信号密度（组合层摊薄诊断）
pos_cnt = {}
for code, ft in feats.items():
    rw = _raw[code]
    c = rw["c"].astype(np.float64); o = rw["o"].astype(np.float64); dates = ft["date"]
    n = len(c)
    for i in np.where(ft["sig"])[0]:
        ds = str(dates[i])
        if not (D0 <= ds <= D1): continue
        e = i+1; x = i+20
        if e >= n or x >= n or o[e] <= 0: continue
        for t in range(e, x+1):
            dd = str(dates[t])
            if dd in didx: pos_cnt[dd] = pos_cnt.get(dd, 0) + 1
cnts = np.array(list(pos_cnt.values())) if pos_cnt else np.array([0])
R["density"] = {"mean_pos": float(cnts.mean()), "max_pos": int(cnts.max()),
                "median_pos": float(np.median(cnts)), "n_days": int(len(cnts))}
print("日均活跃仓位 %.0f (峰值 %d)" % (cnts.mean(), cnts.max()))

# ---------- 8. 稳健性 OFAT ----------
def desc2(sel): return desc([r["t1_20"] for r in sel])
ofat = {}
ofat["基准(生产口径)"] = desc2(rows)
ofat["score>=7"] = desc2([r for r in rows if r["score"] >= 7])
ofat["score>=8"] = desc2([r for r in rows if r["score"] >= 8])
ofat["仅第1砖"] = desc2([r for r in rows if r["consec"] == 1])
ofat["仅第2砖"] = desc2([r for r in rows if r["consec"] == 2])
ofat["dist_low>-1%(贴线)"] = desc2([r for r in rows if r["dist_low"] > -1])
ofat["dist_low<=-1%(深探)"] = desc2([r for r in rows if r["dist_low"] <= -1])
ofat["dist_low<=-3%(深探)"] = desc2([r for r in rows if r["dist_low"] <= -3])
ofat["仅阳线"] = desc2([r for r in rows if r["bullish"]])
ofat["仅站上MA60"] = desc2([r for r in rows if r["above_ma60"]])
ofat["chg20d>0(近20日涨)"] = desc2([r for r in rows if r["chg20d"] > 0])
ofat["chg20d<=0(近20日跌)"] = desc2([r for r in rows if r["chg20d"] <= 0])
R["ofat"] = ofat

R["regime"] = {
    "pre20>0(近期强势)": desc2([r for r in rows if r["pre20"] > 0]),
    "pre20<=0(近期弱势)": desc2([r for r in rows if r["pre20"] <= 0]),
    "vol20<3%(低波)": desc2([r for r in rows if r["vol20"] < 3]),
    "vol20>=3%(高波)": desc2([r for r in rows if r["vol20"] >= 3]),
}

# ---------- 9. 微观 ----------
R["counterparty"] = {
    "pre20": layer(rows, lambda r: ("<0%" if r["pre20"] < 0 else "0-10%" if r["pre20"] < 10 else "10-20%" if r["pre20"] < 20 else "20-40%" if r["pre20"] < 40 else ">=40%")),
    "pre60": layer(rows, lambda r: ("<0%" if r["pre60"] < 0 else "0-20%" if r["pre60"] < 20 else "20-50%" if r["pre60"] < 50 else "50-100%" if r["pre60"] < 100 else ">=100%")),
    "dd20": layer(rows, lambda r: ("<-15%" if r["dd20"] < -15 else "-15~-10%" if r["dd20"] < -10 else "-10~-5%" if r["dd20"] < -5 else ">=-5%")),
    "vol20": layer(rows, lambda r: ("<2%" if r["vol20"] < 2 else "2-3%" if r["vol20"] < 3 else "3-4%" if r["vol20"] < 4 else ">=4%")),
}
# 机制分层（砖位/支撑距离/VAR6A）
R["quality"] = {
    "砖位": layer(rows, lambda r: ("第1砖" if r["consec"] == 1 else "第2砖" if r["consec"] == 2 else "其他")),
    "支撑距离dist_low": layer(rows, lambda r: ("<=-3%" if r["dist_low"] <= -3 else "-3~-1%" if r["dist_low"] <= -1 else "-1~0%" if r["dist_low"] <= 0 else ">0%")),
    "VAR6A水平": layer(rows, lambda r: ("<20" if r["var6a"] < 20 else "20-40" if r["var6a"] < 40 else "40-60" if r["var6a"] < 60 else ">=60")),
    "chg20d": layer(rows, lambda r: ("<-10%" if r["chg20d"] < -10 else "-10~0%" if r["chg20d"] < 0 else "0-5%" if r["chg20d"] < 5 else "5-10%" if r["chg20d"] < 10 else ">=10%")),
}
mae20 = np.array([r["mae20"] for r in rows]); mfe20 = np.array([r["mfe20"] for r in rows])
f5 = np.array([r["first5_min"] for r in rows]); l15 = np.array([r["last15_max"] for r in rows])
bd = np.array([r["below_days"] for r in rows])
R["path"] = {
    "mae20": {"median": float(np.median(mae20)), "mean": float(mae20.mean())},
    "mfe20": {"median": float(np.median(mfe20)), "mean": float(mfe20.mean())},
    "first5_min": {"median": float(np.median(f5)), "mean": float(f5.mean())},
    "last15_max": {"median": float(np.median(l15)), "mean": float(l15.mean())},
    "below_days": {"median": float(np.median(bd)), "mean": float(bd.mean())},
}

mon = defaultdict(list)
for r in rows: mon[r["date"][:6]].append(r["t1_20"])
R["monthly"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(mon.items())}

json.dump(R, open(os.path.join(HERE, "banzhuan_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> banzhuan_results.json")

# 打印摘要
g = R["groups"]
print("\n=== 分组 ===")
for k in ["全体","训练期","验证期","池内","池外","第1砖","第2砖","score=5-6","score=7","score>=8","阳线(C>O)","阴线","站上MA60"]:
    if g.get(k): print("%-16s n=%6d mean=%+.2f%% med=%+.2f%% win=%.1f%%" % (k, g[k]["n"], g[k]["mean"], g[k]["median"], g[k]["win"]))
print("\n=== IC ===")
for k, v in R["ic"].items(): print("%s: mean=%+.4f icir=%+.3f t=%+.2f pos=%.1f%%" % (k, v["mean"], v["icir"], v["t"], v["pos_ratio"]))
print("\n=== 组合(月度cohort) ===")
for k, v in R["port"].items():
    a = v["all"]
    if a: print("%s: total=%+.1f%% sharpe=%+.2f maxdd=%.1f%% train=%+.1f%% valid=%+.1f%%" % (
        k, a["total"], a["sharpe"], a["maxdd"], v["train"]["total"] if v["train"] else 0, v["valid"]["total"] if v["valid"] else 0))
print("\n基准: 等权全A all=%+.1f%% 等权池内 all=%+.1f%%" % (
    R["bench_port"]["等权全A"]["all"]["total"], R["bench_port"]["等权池内"]["all"]["total"]))
print("MAE中位=%.2f%% 超8%%止损占比=%.1f%%" % (R["mae_mismatch"]["median"], R["mae_mismatch"]["frac_below"]["8"]))
