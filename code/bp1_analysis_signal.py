# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step2 信号层检验
口径与B1/B2一致: T日信号收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20), 及收盘价口径fwd20
bp1 = Kimi买点一(平台突破买点): L0 N类门控(n_raw>=40) + 技术五条件
核心归因: ①N类门控增量(tech-only vs 完整信号) ②n_raw分层 ③池内外 ④训练/验证
"""
import pickle, os, json, math, statistics as st, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
print("宇宙 %d 只, 池 %d 只" % (len(feats), len(pool_codes)))

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    srt = np.sort(v); top5 = srt[int(len(srt)*0.95):]
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std()),
            "top5_share": float(100*top5.sum()/v.sum()) if v.sum() != 0 else None}

R = {}

# ---------- 1. bp1信号表 ----------
rows = []
tech_only = []   # 技术条件过但L0否(对照: N类门控的增量价值)
for code, ft in feats.items():
    dates = ft["date"]
    sig = ft["bp1_sig"]; tech = ft["bp1_tech"]; l0 = ft["l0_pass"]
    n_raw = ft["n_raw"]; t1 = ft["t1_20"]; fw = ft["fwd20"]
    for i in np.where(sig | (tech & ~l0))[0]:
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        rec = {"code": code, "date": ds, "is_sig": bool(sig[i]),
               "n_raw": int(n_raw[i]), "in_pool": 1 if code in pool_codes else 0,
               "l2_pass": bool(ft["l2_pass"][i]),
               "d1": float(ft["d1"][i]), "d2": float(ft["d2"][i]), "d3": float(ft["d3"][i]),
               "d4": float(ft["d4"][i]), "d6": float(ft["d6"][i]), "rps": float(ft["rps"][i]),
               "stage_mult": float(ft["stage_mult"][i]), "ext": float(ft["ext"][i]) if not np.isnan(ft["ext"][i]) else None,
               "trend_age": float(ft["trend_age"][i]) if not np.isnan(ft["trend_age"][i]) else None,
               "t1_20": float(t1[i])*PCT, "fwd20": float(fw[i])*PCT,
               "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
               "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
               "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
               "below_days": float(ft["below_days"][i]),
               "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
               "vol_ratio": float(ft["vol_ratio"][i]) if not np.isnan(ft["vol_ratio"][i]) else None,
               "vr5_1": float(ft["vr5_1"][i]) if not np.isnan(ft["vr5_1"][i]) else None,
               "vr_long": float(ft["vr_long"][i]) if not np.isnan(ft["vr_long"][i]) else None,
               "breakout_amp": float(ft["breakout_amp"][i]), "platform_range": float(ft["platform_range"][i]),
               "vol_atr_ratio": float(ft["vol_atr_ratio"][i]), "vp_corr": float(ft["vp_corr"][i]) if not np.isnan(ft["vp_corr"][i]) else None,
               "rsi": float(ft["rsi"][i]) if not np.isnan(ft["rsi"][i]) else None,
               "ma20_slope": float(ft["ma20_slope"][i]) if not np.isnan(ft["ma20_slope"][i]) else None}
        if sig[i]: rows.append(rec)
        else: tech_only.append(rec)
print("bp1信号 %d 条 | 技术过-L0否(对照) %d 条" % (len(rows), len(tech_only)))

groups = {"全体": rows,
          "训练期": [r for r in rows if period_of(r["date"]) == "训练期"],
          "验证期": [r for r in rows if period_of(r["date"]) == "验证期"],
          "池内": [r for r in rows if r["in_pool"]],
          "池外": [r for r in rows if not r["in_pool"]],
          "池内-训练": [r for r in rows if r["in_pool"] and period_of(r["date"]) == "训练期"],
          "池内-验证": [r for r in rows if r["in_pool"] and period_of(r["date"]) == "验证期"],
          "池外-训练": [r for r in rows if not r["in_pool"] and period_of(r["date"]) == "训练期"],
          "池外-验证": [r for r in rows if not r["in_pool"] and period_of(r["date"]) == "验证期"],
          "n_raw 40-49": [r for r in rows if 40 <= r["n_raw"] < 50],
          "n_raw 50-59": [r for r in rows if 50 <= r["n_raw"] < 60],
          "n_raw >=60": [r for r in rows if r["n_raw"] >= 60],
          "L2通过": [r for r in rows if r["l2_pass"]],
          "L2未过": [r for r in rows if not r["l2_pass"]]}
R["groups"] = {k: desc([r["t1_20"] for r in v]) for k, v in groups.items()}
print("\n信号层(t1_20, T+1开盘→T+20收盘):")
for k in ["全体", "训练期", "验证期", "池内", "池外", "池内-训练", "池内-验证", "池外-训练", "池外-验证",
          "n_raw 40-49", "n_raw 50-59", "n_raw >=60", "L2通过", "L2未过"]:
    v = R["groups"][k]
    if v: print("  %-12s n=%6d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%% p10=%+7.1f p90=%+7.1f" % (
        k, v["n"], v["mean"], v["median"], v["win"], v["p10"], v["p90"]))

# ---------- 2. N类门控增量: tech-only vs 完整信号 ----------
R["l0_gate"] = {"完整信号(L0过)": desc([r["t1_20"] for r in rows]),
                "仅技术条件(L0否)": desc([r["t1_20"] for r in tech_only])}
print("\nN类门控增量(同技术条件, L0门槛切分):")
for k, v in R["l0_gate"].items():
    if v: print("  %-18s n=%6d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (k, v["n"], v["mean"], v["median"], v["win"]))

# ---------- 3. n_raw 十分位(信号内部) ----------
allv = np.sort(np.array([r["t1_20"] for r in rows]))
dec = {}
for i in range(10):
    lo, hi = np.percentile(allv, i*10), np.percentile(allv, (i+1)*10)
    seg = allv[(allv >= lo) & (allv <= hi)] if i < 9 else allv[allv >= lo]
    dec["D%d(%.1f~%.1f%%)" % (i+1, lo, hi)] = {"n": int(len(seg)), "share": float(100*len(seg)/len(allv)),
                                               "pnl_share": float(100*seg.sum()/allv.sum()), "mean": float(seg.mean())}
R["deciles"] = dec
print("\n十分位(按t1_20):")
for k, v in dec.items(): print("  %-18s 数量%5.1f%% 盈亏%+7.1f%% 均值%+7.2f%%" % (k, v["share"], v["pnl_share"], v["mean"]))

# ---------- 4. IC(日度截面) ----------
daily = collections.defaultdict(lambda: {"sig": [], "score": [], "fwd": [], "sigfwd": [], "nosigfwd": []})
for code, ft in feats.items():
    dates = ft["date"]; sb = ft["bp1_sig"]; nr_ = ft["n_raw"]; fw = ft["fwd20"]
    for i in range(119, len(dates)-20):
        if np.isnan(fw[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        dd = daily[ds]
        dd["fwd"].append(fw[i]*PCT)
        dd["score"].append(float(nr_[i]))
        if sb[i]: dd["sig"].append(1.0); dd["sigfwd"].append(fw[i]*PCT)
        else: dd["sig"].append(0.0); dd["nosigfwd"].append(fw[i]*PCT)

def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); i = 0; sa = a[order]
    while i < len(a):
        j2 = i
        while j2+1 < len(a) and sa[j2+1] == sa[i]: j2 += 1
        ranks[order[i:j2+1]] = (i+j2)/2.0+1; i = j2+1
    return ranks
def spear(x, y):
    if len(x) < 30: return None
    rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else None

ic = {"sig": [], "score": []}
spread = []
for ds in sorted(daily):
    dd = daily[ds]
    for k in ic:
        xs = dd[k]; ys = dd["fwd"]
        m = ~np.isnan(xs)
        r = spear(np.array(xs)[m], np.array(ys)[m])
        if r is not None: ic[k].append(r)
    if dd["sigfwd"] and dd["nosigfwd"]:
        spread.append(np.mean(dd["sigfwd"])-np.mean(dd["nosigfwd"]))
ic_out = {}
for k, v in ic.items():
    v = np.array(v)
    ic_out[k] = {"n_days": len(v), "mean": float(v.mean()), "std": float(v.std()),
                 "icir": float(v.mean()/v.std()) if v.std() > 0 else None,
                 "pos_ratio": float(100*(v > 0).mean()),
                 "t": float(v.mean()/v.std()*math.sqrt(len(v))) if v.std() > 0 else None}
sp = np.array(spread)
ic_out["spread"] = {"n_days": len(sp), "mean": float(sp.mean()), "pos_ratio": float(100*(sp > 0).mean()),
                    "t": float(sp.mean()/sp.std()*math.sqrt(len(sp))) if sp.std() > 0 else None}
R["ic"] = ic_out
print("\nIC(日度截面, fwd20):")
for k in ["sig", "score", "spread"]:
    v = ic_out[k]
    print("  %-7s days=%d mean=%+.4f icir=%+.3f pos=%5.1f%% t=%+.2f" % (k, v["n_days"], v["mean"], v.get("icir") or -9, v["pos_ratio"], v.get("t") or -9))

# ---------- 5. 朴素对照 ----------
def gm(cond=None):
    vals = []
    for code, ft in feats.items():
        arr = ft["t1_20"]; dates = ft["date"]
        for i in range(119, len(arr)):
            if np.isnan(arr[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            if cond and not cond(ft, i): continue
            vals.append(float(arr[i])*PCT)
    return desc(vals)

naive = {}
naive["bp1信号(基准)"] = desc([r["t1_20"] for r in rows])
naive["仅技术条件不过L0"] = desc([r["t1_20"] for r in tech_only])
naive["全股票等权基准"] = gm()
naive["池内全股票等权基准"] = gm(lambda ft, i: True)  # 占位, 下面按池重算
pool_vals = []
for code, ft in feats.items():
    if code not in pool_codes: continue
    for i in range(119, len(ft["t1_20"])):
        if np.isnan(ft["t1_20"][i]): continue
        ds = ft["date"][i].replace("-", "")
        if D0 <= ds <= D1: pool_vals.append(float(ft["t1_20"][i])*PCT)
naive["池内全股票等权基准"] = desc(pool_vals)
R["naive"] = naive
print("\n朴素对照:")
for k, v in naive.items():
    if v: print("  %-24s n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (k, v["n"], v["mean"], v["median"], v["win"]))

# ---------- 6. 技术五条件单项通过率/增量 ----------
cond_stats = {}
for cname, ckey in [("平台振幅<25%", "platform_range"), ("ATR/MA<0.06", "vol_atr_ratio"),
                    ("突破>3%", "breakout_amp"), ("量比>1.5(5日)", "vr5_1"), ("量比>1.5(60日)", "vr_long")]:
    if cname == "平台振幅<25%":
        v = [r for r in rows if r[ckey] < 25]
    elif cname == "ATR/MA<0.06":
        v = [r for r in rows if r[ckey] < 0.06]
    elif cname == "突破>3%":
        v = [r for r in rows if r[ckey] >= 3]
    else:
        v = [r for r in rows if r[ckey] is not None and r[ckey] > 1.5]
    cond_stats[cname] = desc([r["t1_20"] for r in v])
R["cond_stats"] = cond_stats
print("\n技术条件满足度(信号集内部, 应全部满足=100%):")
for k, v in cond_stats.items():
    if v: print("  %-16s n=%6d (%.1f%% of 信号)" % (k, v["n"], 100*v["n"]/len(rows)))

monthly = collections.Counter(r["date"][:6] for r in rows)
R["monthly_counts"] = dict(monthly)
print("\n月度信号数:", dict(sorted(monthly.items())))

json.dump(R, open(os.path.join(HERE, "bp1_signal_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import csv
with open(os.path.join(HERE, "bp1_signals_flat.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(HERE, "bp1_techonly_flat.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tech_only[0].keys())); w.writeheader(); w.writerows(tech_only)
print("\nsaved -> bp1_signal_results.json / bp1_signals_flat.csv / bp1_techonly_flat.csv")
