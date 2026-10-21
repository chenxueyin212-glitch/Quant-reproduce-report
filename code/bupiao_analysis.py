# -*- coding: utf-8 -*-
"""
补票（单针下20）五项检验 — 信号层/组合层/NAV/稳健性/微观
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20)，训练<20250908，费率0.30%/往返
"""
import pickle, os, json
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "bupiao_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = d["pool_codes"]
# 【D29 修复】日期格式归一：混入的 YYYY-MM-DD 连字符日期会进幽灵月桶（如 "2025-0"）并插错复利顺序
for _ft in feats.values():
    _ft["date"] = np.array([str(x).replace("-", "") for x in _ft["date"]])
print("宇宙 %d 只, 池 %d 只" % (len(feats), len(pool_codes)))

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std())}

def layer(vals, split_by):
    """vals: list of dict; split_by: fn(rec)->key"""
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
                     "ds": float(ft["ds"][i]), "dl": float(ft["dl"][i]),
                     "score": int(ft["score"][i]), "btype": int(ft["bupiao_type"][i]),
                     "chg20d": float(ft["chg20d"][i]),
                     "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
                     "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
                     "below_days": float(ft["below_days"][i]),
                     "first5_min": float(ft["first5_min"][i])*PCT,
                     "last15_max": float(ft["last15_max"][i])*PCT})
print("有效信号 %d 条" % len(rows))

# 右截断统计
n_raw = sum(int(ft["sig"].sum()) for ft in feats.values())
R["truncation"] = {"n_total": n_raw, "n_truncated": n_raw - len(rows)}

# ---------- 1. 分组 ----------
def grp(keyfn):
    g = defaultdict(list)
    for r in rows: g[keyfn(r)].append(r["t1_20"])
    return {k: desc(v) for k, v in g.items()}

groups = {}
groups["全体"] = desc([r["t1_20"] for r in rows])
groups["训练期"] = desc([r["t1_20"] for r in rows if r["period"] == "训练期"])
groups["验证期"] = desc([r["t1_20"] for r in rows if r["period"] == "验证期"])
groups["池内"] = desc([r["t1_20"] for r in rows if r["in_pool"]])
groups["池外"] = desc([r["t1_20"] for r in rows if not r["in_pool"]])
for p in ["训练期", "验证期"]:
    for ip, nm in [(True, "池内"), (False, "池外")]:
        groups["%s-%s" % (nm, p)] = desc([r["t1_20"] for r in rows if r["period"] == p and r["in_pool"] == ip])
# score 分层
groups["score=6"] = desc([r["t1_20"] for r in rows if r["score"] == 6])
groups["score=7"] = desc([r["t1_20"] for r in rows if r["score"] == 7])
groups["score>=8"] = desc([r["t1_20"] for r in rows if r["score"] >= 8])
# 类型分层
groups["标准型(dl>=85,ds<30)"] = desc([r["t1_20"] for r in rows if r["btype"] == 1])
groups["激进型(dl>=80,ds<20)"] = desc([r["t1_20"] for r in rows if r["btype"] == 2])
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
daily = defaultdict(list)
for r in rows: daily[r["date"]].append(r)
ics = []
for ds, lst in daily.items():
    if len(lst) < 3: continue
    t = np.array([x["t1_20"] for x in lst])
    for fac in ["score", "dl", "ds"]:
        f = np.array([x[fac] for x in lst])
        if np.std(f) == 0: continue
        # score 越高越好为正IC；ds 越低(超卖)越好 → 预期负IC
        from scipy.stats import spearmanr
        ic, _ = spearmanr(f, t)
        ics.append((ds, fac, ic))
R["ic"] = {}
for fac in ["score", "dl", "ds"]:
    arr = np.array([x[2] for x in ics if x[1] == fac and not np.isnan(x[2])])
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
    "补票信号": desc([r["t1_20"] for r in rows]),
    "全宇宙等权(同持有期)": desc(base["all"]),
    "全宇宙-训练": desc(base["train"]),
    "全宇宙-验证": desc(base["valid"]),
}

# ---------- 5. 组合层(月度cohort) ----------
def cohort(sim_rows, stop=None):
    """sim_rows: rows with t1_20 (无止损)。stop: 百分比止损(用mae20近似止损触发, t1_20若被截断用stop)"""
    mon = defaultdict(list)
    for r in sim_rows:
        pnl = r["t1_20"]
        if stop is not None and r["mae20"] <= -stop:
            pnl = -stop
        mon[r["date"][:6]].append(pnl)
    # 月度等权
    mrets = []
    for m in sorted(mon):
        mrets.append(np.mean(mon[m]))
    mrets = np.array(mrets)
    if len(mrets) == 0: return None
    nav = np.cumprod(1 + mrets/100.0)
    total = float((nav[-1]-1)*100)
    ann = float(((nav[-1])**(12.0/len(mrets))-1)*100) if len(mrets) > 0 else None
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

# 池内 cohort
R["pool_port"] = {
    "不止损": {"all": cohort([r for r in rows if r["in_pool"]]),
               "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"]),
               "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"])},
    "止损8%": {"all": cohort([r for r in rows if r["in_pool"]], 8),
               "train": cohort([r for r in rows if r["in_pool"] and r["period"] == "训练期"], 8),
               "valid": cohort([r for r in rows if r["in_pool"] and r["period"] == "验证期"], 8)},
}

# 基准 cohort (等权全A/池内)
bench_rows_all = []
bench_rows_pool = []
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

# ---------- 7. NAV (日度等权再平衡) ----------
# 简化: 用月度cohort NAV 已在 port; 这里做日度近似——按信号日构建日收益序列
# 为与前几章可比, 用月度 NAV 的 monthly 字段直接对比即可, 不再重复建日度
R["nav_note"] = "本章主口径=月度cohort等权NAV(同B1/B2报告4.2节口径); 日度等权再平衡对补票高频重叠信号意义有限"

# ---------- 8. 稳健性 OFAT ----------
def sig_filter(fn):
    return [r for r in rows if fn(r)]
ofat = {}
ofat["基准(score>=6,生产口径)"] = desc([r["t1_20"] for r in rows])
# 注: 补票的 OFAT 用信号生成参数无法在结果层重算(掩码依赖原始序列)，
# 改用 score/类型/强度 门槛 + 持有期近似 作为敏感性
ofat["score>=7"] = desc([r["t1_20"] for r in rows if r["score"] >= 7])
ofat["score>=8"] = desc([r["t1_20"] for r in rows if r["score"] >= 8])
ofat["仅标准型"] = desc([r["t1_20"] for r in rows if r["btype"] == 1])
ofat["仅激进型"] = desc([r["t1_20"] for r in rows if r["btype"] == 2])
ofat["dl>=90"] = desc([r["t1_20"] for r in rows if r["dl"] >= 90])
ofat["ds<20"] = desc([r["t1_20"] for r in rows if r["ds"] < 20])
ofat["ds<15"] = desc([r["t1_20"] for r in rows if r["ds"] < 15])
ofat["chg20d>0(顺20日)"] = desc([r["t1_20"] for r in rows if r["chg20d"] > 0])
ofat["chg20d<0(逆20日)"] = desc([r["t1_20"] for r in rows if r["chg20d"] <= 0])
R["ofat"] = ofat

# regime
# 用 pre20 作为状态代理 + vol20
R["regime"] = {
    "pre20>0(近期强势)": desc([r["t1_20"] for r in rows if r["pre20"] > 0]),
    "pre20<=0(近期弱势)": desc([r["t1_20"] for r in rows if r["pre20"] <= 0]),
    "vol20<3%(低波)": desc([r["t1_20"] for r in rows if r["vol20"] < 3]),
    "vol20>=3%(高波)": desc([r["t1_20"] for r in rows if r["vol20"] >= 3]),
}

# ---------- 9. 微观 ----------
# 对手方
R["counterparty"] = {
    "pre20": layer(rows, lambda r: ("<0%" if r["pre20"] < 0 else "0-10%" if r["pre20"] < 10 else "10-20%" if r["pre20"] < 20 else "20-40%" if r["pre20"] < 40 else ">=40%")),
    "pre60": layer(rows, lambda r: ("<0%" if r["pre60"] < 0 else "0-20%" if r["pre60"] < 20 else "20-50%" if r["pre60"] < 50 else "50-100%" if r["pre60"] < 100 else ">=100%")),
    "dd20": layer(rows, lambda r: ("<-15%" if r["dd20"] < -15 else "-15~-10%" if r["dd20"] < -10 else "-10~-5%" if r["dd20"] < -5 else ">=-5%")),
    "vol20": layer(rows, lambda r: ("<2%" if r["vol20"] < 2 else "2-3%" if r["vol20"] < 3 else "3-4%" if r["vol20"] < 4 else ">=4%")),
}
# ds/dl 质量分层
R["quality"] = {
    "ds(短线超卖度)": layer(rows, lambda r: ("<10" if r["ds"] < 10 else "10-20" if r["ds"] < 20 else "20-30" if r["ds"] < 30 else ">=30")),
    "dl(长线趋势度)": layer(rows, lambda r: ("80-85" if r["dl"] < 85 else "85-90" if r["dl"] < 90 else "90-95" if r["dl"] < 95 else ">=95")),
    "chg20d": layer(rows, lambda r: ("<-10%" if r["chg20d"] < -10 else "-10~0%" if r["chg20d"] < 0 else "0-5%" if r["chg20d"] < 5 else "5-10%" if r["chg20d"] < 10 else ">=10%")),
}
# 路径
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

# 月度分布
mon = defaultdict(list)
for r in rows: mon[r["date"][:6]].append(r["t1_20"])
R["monthly"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(mon.items())}

json.dump(R, open(os.path.join(HERE, "bupiao_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> bupiao_results.json")

# 打印摘要
g = R["groups"]
print("\n=== 分组 ===")
for k in ["全体","训练期","验证期","池内","池外","score=6","score=7","score>=8","标准型(dl>=85,ds<30)","激进型(dl>=80,ds<20)"]:
    if g.get(k): print("%-24s n=%6d mean=%+.2f%% med=%+.2f%% win=%.1f%%" % (k, g[k]["n"], g[k]["mean"], g[k]["median"], g[k]["win"]))
print("\n=== IC ===")
for k, v in R["ic"].items(): print("%s: mean=%+.4f icir=%+.3f t=%+.2f pos=%.1f%%" % (k, v["mean"], v["icir"], v["t"], v["pos_ratio"]))
print("\n=== 组合(月度cohort) ===")
for k, v in R["port"].items():
    a = v["all"]
    if a: print("%s: total=%+.1f%% sharpe=%+.2f maxdd=%.1f%% train=%+.1f%% valid=%+.1f%%" % (
        k, a["total"], a["sharpe"], a["maxdd"], v["train"]["total"] if v["train"] else 0, v["valid"]["total"] if v["valid"] else 0))
print("\n基准: 等权全A all=%+.1f%% 等权池内 all=%+.1f%%" % (
    R["bench_port"]["等权全A"]["all"]["total"], R["bench_port"]["等权池内"]["all"]["total"]))
print("\nMAE中位=%.2f%% 超8%%止损占比=%.1f%%" % (R["mae_mismatch"]["median"], R["mae_mismatch"]["frac_below"]["8"]))
