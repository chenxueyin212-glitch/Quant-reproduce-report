# -*- coding: utf-8 -*-
"""
B1因子重构评估 — Step5 微观经济假设检验
用户假设:
  H1 真正走趋势的票不会轻易结束(趋势持续) — 检验: 趋势强度分层
  H2 适度回调反而是机会 — 检验: 回调深度分层(倒U?)
  H3 赚被洗走的人跟风上车的人的钱 — 检验: 洗盘路径/量能跟风代理
  被谁赚钱: 更早参与趋势的人 / 认定趋势结束的人 — 检验: 前期涨幅透支度/dc-vs-dk阶段
输出: b1_micro_results.json
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

def dstat(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) == 0: return None
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

def quintile_report(rows, key, label, base_rows=None, base_key=None):
    """rows: list of dict; 按 key 分五组, 报各组 forward; base_rows 可选(同分层的基准)"""
    vals = sorted(r[key] for r in rows if not np.isnan(r[key]))
    if len(vals) < 100: return None
    qs = [np.percentile(vals, q) for q in [0, 20, 40, 60, 80, 100]]
    out = {}
    for i in range(5):
        lo, hi = qs[i], qs[i+1]
        sub = [r for r in rows if not np.isnan(r[key]) and lo <= r[key] <= (hi if i == 4 else hi)]
        s = dstat([r["t1_20"] for r in sub])
        b = None
        if base_rows is not None:
            bsub = [r for r in base_rows if not np.isnan(r[base_key]) and lo <= r[base_key] <= (hi if i == 4 else hi)]
            b = dstat([r["t1_20"] for r in bsub])
        out["Q%d(%.1f~%.1f)" % (i+1, lo, hi)] = {"sig": s, "base": b,
                                                 "rel": (s["mean"]-b["mean"]) if (s and b) else None}
    return out

# ---------- 收集信号与全宇宙后视特征 ----------
sigs = []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        t = "b1_dc" if ft["sig_dc"][i] else "b1_dk"
        sigs.append({"code": code, "date": ds, "type": t, "i": int(i),
                     "in_pool": 1 if code in pool_codes else 0,
                     "t1_20": float(ft["t1_20"][i])*PCT, "fwd20": float(ft["fwd20"][i])*PCT,
                     "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
                     "dd20": float(ft["dd20"][i])*PCT, "j": float(ft["j"][i]),
                     "depth": float(ft["depth_zt"][i] if t == "b1_dc" else ft["depth_zd"][i])*PCT,
                     "vol20": float(ft["vol20"][i])*PCT,
                     "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
                     "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
                     "below_days": float(ft["below_days"][i]), "vol_ratio": float(ft["vol_ratio"][i]),
                     "above_ma114": float(ft["c"][i]/ft["ma114"][i]-1)*PCT if not np.isnan(ft["ma114"][i]) else np.nan,
                     "zt_slope": float(ft["zt"][i]/ft["zt"][i-20]-1)*PCT if i >= 20 and not np.isnan(ft["zt"][i]) else np.nan})
sigs = [r for r in sigs if not np.isnan(r["t1_20"])]
print("有效信号: %d" % len(sigs))

# 全宇宙后视(用于同分层基准)
universe = []
for code, ft in feats.items():
    dates = ft["date"]
    t1 = ft["t1_20"]; p60 = ft["pre60"]; p20 = ft["pre20"]
    for i in range(59, len(t1)-20):
        if np.isnan(t1[i]) or np.isnan(p60[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        universe.append({"pre60": float(p60[i])*PCT, "pre20": float(p20[i])*PCT, "t1_20": float(t1[i])*PCT})
print("宇宙样本: %d" % len(universe))

R = {}

# ---------- H1: 趋势强度分层 ----------
R["H1_pre60"] = quintile_report(sigs, "pre60", "前60日涨幅", universe, "pre60")
print("\nH1 趋势强度(前60日涨幅)分层:")
for k, v in R["H1_pre60"].items():
    s, b = v["sig"], v["base"]
    print("  %-14s 信号n=%6d mean=%+6.2f%% win%5.1f%% | 同层基准n=%7d mean=%+6.2f%% | rel=%+6.2f" % (
        k, s["n"], s["mean"], s["win"], b["n"], b["mean"], v["rel"]))
R["H1_above_ma114"] = {}
for lab, cond in [("低于MA114(<-5%)", lambda r: r["above_ma114"] < -5),
                  ("接近MA114(-5~+5%)", lambda r: -5 <= r["above_ma114"] <= 5),
                  ("高于MA114(>5%)", lambda r: r["above_ma114"] > 5),
                  ("高于MA114(>20%)", lambda r: r["above_ma114"] > 20)]:
    sub = [r for r in sigs if cond(r)]
    s = dstat([r["t1_20"] for r in sub])
    R["H1_above_ma114"][lab] = {"sig": s, "base": None}
    if s: print("  %-18s 信号n=%6d mean=%+6.2f%% win%5.1f%%" % (lab, s["n"], s["mean"], s["win"]))

# ---------- H2: 回调深度分层 ----------
R["H2_dd20"] = quintile_report(sigs, "dd20", "信号日20日回撤深度")
print("\nH2 回调深度(信号日最低/20日高)分层:")
for k, v in R["H2_dd20"].items():
    s = v["sig"]
    print("  %-16s 信号n=%6d mean=%+6.2f%% median=%+6.2f%% win%5.1f%%" % (k, s["n"], s["mean"], s["median"], s["win"]))
R["H2_j"] = {}
jb = [(-100, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
for lo, hi in jb:
    sub = [r for r in sigs if lo <= r["j"] < hi]
    s = dstat([r["t1_20"] for r in sub])
    R["H2_j"]["J(%d,%d)" % (lo, hi)] = s
    if s: print("  J∈[%d,%d) n=%6d mean=%+6.2f%% win%5.1f%%" % (lo, hi, s["n"], s["mean"], s["win"]))
R["H2_depth"] = quintile_report(sigs, "depth", "穿线深度")
print("\nH2 穿线深度(最低价穿破趋势线的幅度)分层:")
for k, v in R["H2_depth"].items():
    s = v["sig"]
    print("  %-16s 信号n=%6d mean=%+6.2f%% win%5.1f%%" % (k, s["n"], s["mean"], s["win"]))

# ---------- H3: 洗盘路径 ----------
paths = {"洗盘后上行": lambda r: r["first5_min"] < -2 and r["last15_max"] > 3 and r["t1_20"] > 0,
         "直线上攻": lambda r: r["first5_min"] >= -2 and r["t1_20"] > 0,
         "阴跌破位": lambda r: r["t1_20"] < 0 and r["first5_min"] < -2,
         "高开回落": lambda r: r["mfe20"] > 8 and r["t1_20"] < 0,
         "其他": lambda r: True}
R["H3_paths"] = {}
cnt = {}
for k, f in paths.items():
    if k == "其他": continue
    cnt[k] = sum(1 for r in sigs if f(r))
assigned = sum(cnt.values())
cnt["其他"] = len(sigs) - assigned
print("\nH3 持仓路径分布(前5日最低 vs 后15日最高):")
for k, v in cnt.items():
    sub = [r for r in sigs if paths[k](r)] if k != "其他" else [r for r in sigs if not any(paths[kk](r) for kk in cnt if kk != "其他")]
    s = dstat([r["t1_20"] for r in sub])
    R["H3_paths"][k] = {"n": v, "share": 100*v/len(sigs), "stat": s}
    if s: print("  %-8s 占比%5.1f%% n=%6d mean=%+6.2f%% median=%+6.2f%% win%5.1f%%" % (k, 100*v/len(sigs), s["n"], s["mean"], s["median"], s["win"]))
# 先破后立频率: 曾跌破入场价但最终收正
dip_then_up = [r for r in sigs if r["first5_min"] < 0 and r["t1_20"] > 0]
never_dip = [r for r in sigs if r["first5_min"] >= 0 and r["t1_20"] > 0]
R["H3_dip_then_up"] = {"先跌后涨": dstat([r["t1_20"] for r in dip_then_up]),
                       "不跌直涨": dstat([r["t1_20"] for r in never_dip])}
print("  先跌后涨: n=%d mean=%+.2f%% | 不跌直涨: n=%d mean=%+.2f%%" % (
    len(dip_then_up), np.mean([r["t1_20"] for r in dip_then_up]) if dip_then_up else 0,
    len(never_dip), np.mean([r["t1_20"] for r in never_dip]) if never_dip else 0))

# ---------- H3b: 量能跟风代理 ----------
R["H3_volratio"] = quintile_report(sigs, "vol_ratio", "后20日/前20日均量")
print("\nH3b 量能比(信号后/前20日均量)分层:")
for k, v in R["H3_volratio"].items():
    s = v["sig"]
    print("  %-16s 信号n=%6d mean=%+6.2f%% win%5.1f%%" % (k, s["n"], s["mean"], s["win"]))

# ---------- 被谁赚钱: 透支度/dc-vs-dk ----------
R["ext_pre20"] = {}
for lo, hi in [(-100, 0), (0, 10), (10, 20), (20, 30), (30, 100)]:
    sub = [r for r in sigs if lo <= r["pre20"] < hi]
    s = dstat([r["t1_20"] for r in sub])
    R["ext_pre20"]["pre20[%d,%d)" % (lo, hi)] = s
    if s: print("  前期20日涨幅∈[%d,%d)%% n=%6d mean=%+6.2f%% win%5.1f%%" % (lo, hi, s["n"], s["mean"], s["win"]))
R["dc_vs_dk"] = {}
for t in ["b1_dc", "b1_dk"]:
    sub = [r for r in sigs if r["type"] == t]
    s = dstat([r["t1_20"] for r in sub])
    s2 = dstat([r["pre60"] for r in sub])
    R["dc_vs_dk"][t] = {"fwd": s, "pre60": s2}
    print("  %s n=%6d mean=%+6.2f%% win%5.1f%% | 信号日前60涨幅中位%+6.2f%%" % (t, s["n"], s["mean"], s["win"], s2["median"]))

json.dump(R, open(os.path.join(HERE, "b1_micro_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> b1_micro_results.json")
