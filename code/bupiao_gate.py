# -*- coding: utf-8 -*-
"""
A级修复验证：补票三重硬门禁重测
门禁: ① chg20d>0（剔逆势补票） ② pre60<50%（剔接盘） ③ dd20>-15%（剔价值陷阱）
输出: 各门禁单开/双开/三开的边际效果 + 训练/验证 + 池内外 + cohort 组合对照
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场，训练<20250908
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
for _ft in feats.values():
    _ft["date"] = np.array([str(x).replace("-", "") for x in _ft["date"]])
print("宇宙 %d 只, 池 %d 只" % (len(feats), len(pool_codes)))

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90))}

# ================= 信号表（同 bupiao_analysis.py） =================
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
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT})
print("有效信号 %d 条" % len(rows))

G = {
    "chg20d>0": lambda r: r["chg20d"] > 0,
    "pre60<50%": lambda r: r["pre60"] < 50,
    "dd20>-15%": lambda r: r["dd20"] > -15,
}

def apply_gates(gate_names):
    return [r for r in rows if all(G[n](r) for n in gate_names)]

def stats_block(rs):
    out = {
        "全体": desc([r["t1_20"] for r in rs]),
        "训练期": desc([r["t1_20"] for r in rs if r["period"] == "训练期"]),
        "验证期": desc([r["t1_20"] for r in rs if r["period"] == "验证期"]),
        "池内": desc([r["t1_20"] for r in rs if r["in_pool"]]),
        "池外": desc([r["t1_20"] for r in rs if not r["in_pool"]]),
        "池内-训练期": desc([r["t1_20"] for r in rs if r["in_pool"] and r["period"] == "训练期"]),
        "池内-验证期": desc([r["t1_20"] for r in rs if r["in_pool"] and r["period"] == "验证期"]),
        "池外-训练期": desc([r["t1_20"] for r in rs if not r["in_pool"] and r["period"] == "训练期"]),
        "池外-验证期": desc([r["t1_20"] for r in rs if not r["in_pool"] and r["period"] == "验证期"]),
    }
    return out

R = {"n_base": len(rows)}

# ================= 门禁组合矩阵 =================
combos = [
    ("基准(无门禁)", []),
    ("①chg20d>0", ["chg20d>0"]),
    ("②pre60<50%", ["pre60<50%"]),
    ("③dd20>-15%", ["dd20>-15%"]),
    ("①+②", ["chg20d>0", "pre60<50%"]),
    ("①+③", ["chg20d>0", "dd20>-15%"]),
    ("②+③", ["pre60<50%", "dd20>-15%"]),
    ("①+②+③(三重门禁)", ["chg20d>0", "pre60<50%", "dd20>-15%"]),
]
gate_matrix = {}
for name, gns in combos:
    rs = apply_gates(gns)
    blk = stats_block(rs)
    blk["keep_pct"] = 100.0*len(rs)/len(rows)
    gate_matrix[name] = blk
R["gate_matrix"] = gate_matrix

# ================= 三重门禁下的参数加严对照 =================
triple = apply_gates(["chg20d>0", "pre60<50%", "dd20>-15%"])
tight = {}
tight["三重门禁(基准)"] = desc([r["t1_20"] for r in triple])
tight["+dl>=90"] = desc([r["t1_20"] for r in triple if r["dl"] >= 90])
tight["+ds<15"] = desc([r["t1_20"] for r in triple if r["ds"] < 15])
tight["+score>=8"] = desc([r["t1_20"] for r in triple if r["score"] >= 8])
tight["+dl>=90 & ds<15"] = desc([r["t1_20"] for r in triple if r["dl"] >= 90 and r["ds"] < 15])
tight["仅dl>=90(无门禁对照)"] = desc([r["t1_20"] for r in rows if r["dl"] >= 90])
R["tight"] = tight

# ================= cohort 组合 =================
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
            "sharpe": float(mrets.mean()/mrets.std()*np.sqrt(12)) if mrets.std() > 0 else None,
            "maxdd": float(dd.min()), "win_m": float(100*(mrets > 0).mean()),
            "monthly": {m: float(v) for m, v in zip(sorted(mon), mrets)}}

port = {}
for label, rs in [("基准(无门禁)", rows), ("①+②+③(三重门禁)", triple),
                  ("②+③(pre60+dd20)", apply_gates(["pre60<50%", "dd20>-15%"]))]:
    port[label] = {
        "不止损": {"all": cohort(rs),
                   "train": cohort([r for r in rs if r["period"] == "训练期"]),
                   "valid": cohort([r for r in rs if r["period"] == "验证期"])},
        "止损8%": {"all": cohort(rs, 8),
                   "train": cohort([r for r in rs if r["period"] == "训练期"], 8),
                   "valid": cohort([r for r in rs if r["period"] == "验证期"], 8)},
        "池内不止损": {"all": cohort([r for r in rs if r["in_pool"]])},
        "池内止损8%": {"all": cohort([r for r in rs if r["in_pool"]], 8),
                       "valid": cohort([r for r in rs if r["in_pool"] and r["period"] == "验证期"], 8)},
    }
R["port"] = port

# ================= 被剔除组对照 =================
removed = [r for r in rows if not all(G[n](r) for n in ["chg20d>0", "pre60<50%", "dd20>-15%"])]
R["removed"] = {
    "全体": desc([r["t1_20"] for r in removed]),
    "训练期": desc([r["t1_20"] for r in removed if r["period"] == "训练期"]),
    "验证期": desc([r["t1_20"] for r in removed if r["period"] == "验证期"]),
}

json.dump(R, open(os.path.join(HERE, "bupiao_gate_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> bupiao_gate_results.json")

# ================= 摘要 =================
print("\n=== 门禁矩阵（均值/保留率） ===")
for name, _ in combos:
    b = gate_matrix[name]; a = b["全体"]
    if not a: continue
    va = b["验证期"]; tr = b["训练期"]; pv = b["池内-验证期"]
    print("%-18s keep=%5.1f%% n=%5d mean=%+.2f%% med=%+.2f%% win=%.1f%% | tr=%+.2f va=%+.2f | 池内-验=%s" % (
        name, b["keep_pct"], a["n"], a["mean"], a["median"], a["win"],
        tr["mean"] if tr else 0, va["mean"] if va else 0,
        ("%+.2f" % pv["mean"]) if pv else "—"))
print("\n=== 参数加严（三重门禁下） ===")
for k, v in tight.items():
    if v: print("%-22s n=%5d mean=%+.2f%% win=%.1f%%" % (k, v["n"], v["mean"], v["win"]))
print("\n=== cohort ===")
for label in port:
    for stop in ["不止损", "止损8%"]:
        a = port[label][stop]["all"]; va = port[label][stop]["valid"]
        if a: print("%-18s %-8s total=%+7.1f%% sharpe=%+.2f maxdd=%.1f%% | va=%+.1f%%" % (
            label, stop, a["total"], a["sharpe"] or 0, a["maxdd"], va["total"] if va else 0))
print("\n=== 被剔除组 ===")
for k, v in R["removed"].items():
    if v: print("%-6s n=%5d mean=%+.2f%%" % (k, v["n"], v["mean"]))
