# -*- coding: utf-8 -*-
"""
B1因子重构评估 — Step2 信号层有效性检验
口径: T日信号(bar>=59起) → T+1开盘入场 → T+20收盘出场(无止损) = t1_20
     另报 fwd20_c2c(close[T]/close[T+20], 与旧CSV ret20族一致) 与 fwd19_c2c(旧CSV原值)
切分: 训练期 < 20250908 ≤ 验证期 (与既有框架一致)
输出: b1_signal_results.json
"""
import pickle, csv, os, json, collections, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"   # 信号日全窗口(完整前瞻由 t1_20 非NaN保证)

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats, bench = d["feats"], d["bench"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(vals):
    v = np.array(vals, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    srt = np.sort(v)
    top5 = srt[int(len(srt)*0.95):]
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std()),
            "top5_share": float(100*top5.sum()/v.sum()) if v.sum() != 0 else None}

# ---------- 1. 信号表 ----------
PCT = 100.0  # 管道内存的是小数, 统一转百分数
rows = []
for code, ft in feats.items():
    dates = ft["date"]; in_pool = 1 if code in pool_codes else 0
    for i in np.where(ft["sig_base"] > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        t = "b1_dc" if ft["sig_dc"][i] else "b1_dk"
        rows.append({
            "code": code, "date": ds, "type": t, "in_pool": in_pool,
            "t1_20": float(ft["t1_20"][i])*PCT, "fwd20": float(ft["fwd20"][i])*PCT, "fwd19": float(ft["fwd19"][i])*PCT,
            "j": float(ft["j"][i]), "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
            "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
            "depth": float(ft["depth_zt"][i] if t == "b1_dc" else ft["depth_zd"][i])*PCT,
            "dist_line": float(ft["dist_zt"][i] if t == "b1_dc" else ft["dist_zd"][i])*PCT,
            "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
            "below_days": float(ft["below_days"][i]),
            "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
            "vol_ratio": float(ft["vol_ratio"][i]),
            "complete": 1 if not np.isnan(ft["t1_20"][i]) else 0,
        })
print("信号总数(bar>=59, %s~%s): %d, 其中前瞻完整: %d" % (D0, D1, len(rows), sum(r["complete"] for r in rows)))

# ---------- 2. 分组统计 ----------
def group_stats(sub, key="t1_20"):
    return desc([r[key] for r in sub])

groups = {}
groups["全体"] = [r for r in rows if r["complete"]]
groups["训练期"] = [r for r in groups["全体"] if period_of(r["date"]) == "训练期"]
groups["验证期"] = [r for r in groups["全体"] if period_of(r["date"]) == "验证期"]
groups["池内"] = [r for r in groups["全体"] if r["in_pool"]]
groups["池外"] = [r for r in groups["全体"] if not r["in_pool"]]
groups["池内-训练"] = [r for r in groups["池内"] if period_of(r["date"]) == "训练期"]
groups["池内-验证"] = [r for r in groups["池内"] if period_of(r["date"]) == "验证期"]
groups["池外-训练"] = [r for r in groups["池外"] if period_of(r["date"]) == "训练期"]
groups["池外-验证"] = [r for r in groups["池外"] if period_of(r["date"]) == "验证期"]
groups["dc型"] = [r for r in groups["全体"] if r["type"] == "b1_dc"]
groups["dk型"] = [r for r in groups["全体"] if r["type"] == "b1_dk"]
out_groups = {k: group_stats(v) for k, v in groups.items()}
for k, v in out_groups.items():
    if v: print("%-8s n=%6d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%% p10=%+7.2f p90=%+7.2f top5share=%5.1f%%" % (
        k, v["n"], v["mean"], v["median"], v["win"], v["p10"], v["p90"], v["top5_share"] or -999))

# ---------- 3. 分布十分位 + 尾部 ----------
allv = np.sort(np.array([r["t1_20"] for r in groups["全体"]]))
dec = {}
for i in range(10):
    lo, hi = np.percentile(allv, i*10), np.percentile(allv, (i+1)*10)
    seg = allv[(allv >= lo) & (allv <= hi)] if i < 9 else allv[allv >= lo]
    dec["D%d(%.1f~%.1f%%)" % (i+1, lo, hi)] = {"n": int(len(seg)), "share": float(100*len(seg)/len(allv)),
                                               "pnl_share": float(100*seg.sum()/allv.sum()), "mean": float(seg.mean())}
print("\n十分位分布(按收益):")
for k, v in dec.items(): print("  %-18s 数量占比%5.1f%% 盈亏贡献%+7.1f%% 组均值%+7.2f%%" % (k, v["share"], v["pnl_share"], v["mean"]))

# ---------- 4. IC: 日度截面 RankIC ----------
# 收集每日截面 (score, fwd20 / t1_20)
daily = collections.defaultdict(lambda: {"sig": [], "negj": [], "negdz": [], "fwd": [], "t1": [],
                                         "sigfwd": [], "nosigfwd": []})
for code, ft in feats.items():
    dates = ft["date"]; sb = ft["sig_base"]; j = ft["j"]; dz = ft["dist_zt"]
    fw = ft["fwd20"]; t1 = ft["t1_20"]
    for i in range(59, len(dates)-20):
        if np.isnan(fw[i]) or np.isnan(dz[i]) or np.isnan(j[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        dd = daily[ds]
        dd["fwd"].append(fw[i]*PCT); dd["t1"].append(t1[i]*PCT if not np.isnan(t1[i]) else np.nan)
        dd["negj"].append(-j[i]); dd["negdz"].append(-dz[i])
        if sb[i]:
            dd["sig"].append(1.0); dd["sigfwd"].append(fw[i]*PCT)
        else:
            dd["sig"].append(0.0); dd["nosigfwd"].append(fw[i]*PCT)

def rankdata(a):
    """平均秩"""
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); i = 0
    sa = a[order]
    while i < len(a):
        j2 = i
        while j2+1 < len(a) and sa[j2+1] == sa[i]: j2 += 1
        ranks[order[i:j2+1]] = (i+j2)/2.0 + 1
        i = j2+1
    return ranks

def spear(x, y):
    if len(x) < 30: return None
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/denom) if denom > 0 else None

ic = {"sig": [], "negj": [], "negdz": [], "sig_t1": []}
spread_daily = []
for ds in sorted(daily):
    dd = daily[ds]
    for k in ["sig", "negj", "negdz"]:
        r = spear(dd[k], dd["fwd"])
        if r is not None: ic[k].append(r)
    pairs = [(s, t) for s, t in zip(dd["sig"], dd["t1"]) if not np.isnan(t)]
    if len(pairs) >= 30:
        r = spear([p[0] for p in pairs], [p[1] for p in pairs])
        if r is not None: ic["sig_t1"].append(r)
    if dd["sigfwd"] and dd["nosigfwd"]:
        spread_daily.append(np.mean(dd["sigfwd"]) - np.mean(dd["nosigfwd"]))
ic_out = {}
for k, v in ic.items():
    v = np.array(v)
    ic_out[k] = {"n_days": len(v), "mean": float(v.mean()), "std": float(v.std()),
                 "icir": float(v.mean()/v.std()) if v.std() > 0 else None,
                 "pos_ratio": float(100*(v > 0).mean()), "t": float(v.mean()/v.std()*math.sqrt(len(v))) if v.std() > 0 else None}
ic_out = {}
for k, v in ic.items():
    v = np.array(v)
    ic_out[k] = {"n_days": len(v), "mean": float(v.mean()), "std": float(v.std()),
                 "icir": float(v.mean()/v.std()) if v.std() > 0 else None,
                 "pos_ratio": float(100*(v > 0).mean()), "t": float(v.mean()/v.std()*math.sqrt(len(v))) if v.std() > 0 else None}
sp = np.array(spread_daily)
ic_out["spread"] = {"n_days": len(sp), "mean": float(sp.mean()), "median": float(np.median(sp)),
                    "pos_ratio": float(100*(sp > 0).mean()), "std": float(sp.std()),
                    "t": float(sp.mean()/sp.std()*math.sqrt(len(sp))) if sp.std() > 0 else None}
print("\n日度截面RankIC(fwd20, 全股票):")
for k in ["sig", "negj", "negdz", "spread"]:
    v = ic_out[k]
    print("  %-7s days=%d mean=%+.4f icir=%+.3f pos=%5.1f%% t=%+.2f" % (
        k, v["n_days"], v["mean"], v.get("icir") or -999, v["pos_ratio"], v.get("t") or -999))

# ---------- 5. 条件归因 / 朴素基准 ----------
def group_mean(cond_fn, mode="all", top_pct=None, key="t1_20"):
    vals = []
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        arr = ft[key]
        for i in range(59, len(arr)):
            if np.isnan(arr[i]): continue
            ds = ft["date"][i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            if cond_fn(ft, i): vals.append(float(arr[i])*PCT)
    return desc(vals)

naive = {}
naive["B1信号(基准)"] = group_stats(groups["全体"])
naive["J<20单独(超卖反弹)"] = group_mean(lambda ft, i: ft["cond_j"][i] == 1)
naive["触趋势线不收稳单独"] = group_mean(lambda ft, i: ft["hold_zt"][i] == 1)
naive["触多空线不收稳单独"] = group_mean(lambda ft, i: (ft["touch_zd"][i] == 1) and (ft["c"][i] >= ft["zd"][i]*0.99))
# 动量: 每日 prior20 前10%
mom_vals = []
per_day = collections.defaultdict(list)
for code, ft in feats.items():
    p20, fw = ft["pre20"], ft["t1_20"]
    for i in range(59, len(fw)):
        if np.isnan(p20[i]) or np.isnan(fw[i]): continue
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        per_day[ds].append((p20[i], fw[i]))
for ds, lst in per_day.items():
    lst.sort(key=lambda x: -x[0])
    k = max(1, int(len(lst)*0.1))
    mom_vals += [v for _, v in lst[:k]]
naive["前20日收益top10%(动量)"] = desc(mom_vals)
naive["全股票等权基准"] = group_mean(lambda ft, i: True, mode="all")
# 宇宙内基准(同口径同期)
naive["池内全股票等权基准"] = group_mean(lambda ft, i: True, mode="pool")
naive["池外全股票等权基准"] = group_mean(lambda ft, i: True, mode="out")
print("\n条件归因与朴素基准(t1_20, %s~%s):" % (D0, D1))
for k, v in naive.items():
    if v: print("  %-22s n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (k, v["n"], v["mean"], v["median"], v["win"]))
rel = {
    "B1信号-全股票基准": naive["B1信号(基准)"]["mean"] - naive["全股票等权基准"]["mean"],
    "池内B1-池内基准": group_stats(groups["池内"])["mean"] - naive["池内全股票等权基准"]["mean"],
    "池外B1-池外基准": group_stats(groups["池外"])["mean"] - naive["池外全股票等权基准"]["mean"],
    "J<20-基准": naive["J<20单独(超卖反弹)"]["mean"] - naive["全股票等权基准"]["mean"],
    "触趋势线-基准": naive["触趋势线不收稳单独"]["mean"] - naive["全股票等权基准"]["mean"],
}
print("\n相对基准(信号alpha, 百分点):")
for k, v in rel.items(): print("  %-16s %+6.2f pct" % (k, v))

# ---------- 5b. 分期基准 + 旧窗口对齐 ----------
def window_stats(cond_fn, mode, d0, d1, key="t1_20"):
    vals = []
    for code, ft in feats.items():
        if mode == "pool" and code not in pool_codes: continue
        if mode == "out" and code in pool_codes: continue
        arr = ft[key]; dates = ft["date"]
        for i in range(59, len(arr)):
            if np.isnan(arr[i]): continue
            ds = dates[i].replace("-", "")
            if not (d0 <= ds <= d1): continue
            if cond_fn(ft, i): vals.append(float(arr[i])*PCT)
    return desc(vals)

legacy = {}
for pname, (d0, d1) in {"训练期": ("20231101", "20250907"), "验证期": ("20250908", "20260821")}.items():
    legacy[pname] = {
        "B1_全体": window_stats(lambda ft, i: ft["sig_base"][i] == 1, "all", d0, d1),
        "B1_池内": window_stats(lambda ft, i: ft["sig_base"][i] == 1, "pool", d0, d1),
        "B1_池外": window_stats(lambda ft, i: ft["sig_base"][i] == 1, "out", d0, d1),
        "基准_全体": window_stats(lambda ft, i: True, "all", d0, d1),
        "基准_池内": window_stats(lambda ft, i: True, "pool", d0, d1),
        "基准_池外": window_stats(lambda ft, i: True, "out", d0, d1),
    }
    for k, v in legacy[pname].items():
        if v: print("  %s %-8s n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (pname, k, v["n"], v["mean"], v["median"], v["win"]))
# 旧框架窗口 2024-07~2026-07 池内 (对齐既有 +3.34%/53.5%)
lw = window_stats(lambda ft, i: ft["sig_base"][i] == 1, "pool", "20240701", "20260701")
lwb = window_stats(lambda ft, i: True, "pool", "20240701", "20260701")
print("\n旧框架窗口(2024-07~2026-07, 池内): B1 n=%d mean=%+.2f%% win=%.1f%% | 池内基准 mean=%+.2f%% | 相对%+.2f pct" % (
    lw["n"], lw["mean"], lw["win"], lwb["mean"], lw["mean"]-lwb["mean"]))

# ---------- 6. 月度信号计数 ----------
monthly = collections.Counter(r["date"][:6] for r in rows)
print("\n月度信号数(全窗口):")
for ym in sorted(monthly): print("  %s: %d" % (ym, monthly[ym]))

json.dump({"groups": out_groups, "deciles": dec, "ic": ic_out, "naive": naive, "relative": rel,
           "legacy_period": {k: {kk: vv for kk, vv in v.items()} for k, v in legacy.items()},
           "legacy_window_pool": {"b1": lw, "baseline": lwb},
           "monthly_counts": dict(monthly),
           "meta": {"n_signals_total": len(rows), "n_complete": sum(r["complete"] for r in rows),
                    "train_end": TRAIN_END, "window": [D0, D1]}},
          open(os.path.join(HERE, "b1_signal_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 信号明细落盘(供后续微观检验复用)
with open(os.path.join(HERE, "b1_signals_flat.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nsaved -> b1_signal_results.json / b1_signals_flat.csv")
