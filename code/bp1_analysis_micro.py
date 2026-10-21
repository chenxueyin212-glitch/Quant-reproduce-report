# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step5 微观假设检验
bp1机制: T日平台突破(技术五条件) + L0 N类门控(n_raw>=40) → T+1开盘入场 → 20个交易日

bp1微观经济学假设:
  H1(平台质量): 压缩越紧(振幅小/ATR低/突破幅度适中)的平台, 突破后延续越强
  H2(量能确认): 突破放量=新资金; vr5_1(事前5日)判别力 vs vol_ratio(事后跟风)
  H3(赚谁的钱): 突破日追高者 + 平台洗盘期下车者; pre20/pre60/dd20分层
  H4(牛股基因): N类总分与五维度各自的截面判别力
  H5(路径形态): 突破后立即奔跑 vs 先洗盘; MAE发生时点

模块:
  1. 技术五条件质量分层(platform_range/vol_atr_ratio/breakout_amp/vr5_1/vr_long) — H1/H2
  2. N类评分分层(总分 + d1..d6各维度) — H4
  3. 对手方分层(pre20/pre60/dd20) — H3
  4. 路径分析(bp1 vs 全A基准; MAE/MFE/前5后15/below_days; MAE发生时点) — H5
  5. 量能事前性(vr5_1五分位 vs 前5/后15 + vol_ratio) — H2
  6. 入场滑点(T+1跳空) + 收益对数分解(gap/首日/剩余19日)
"""
import pickle, os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
print("loaded %d stocks" % len(feats))

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    srt = np.sort(v); top5 = srt[int(len(srt)*0.95):]
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std()),
            "top5_share": float(100*top5.sum()/v.sum()) if v.sum() != 0 else None}

def pdesc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

def welch(a, b):
    a = np.array(a, float); b = np.array(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 30 or len(b) < 30: return None
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = math.sqrt(va+vb)
    if se == 0: return None
    t = (a.mean()-b.mean())/se
    p = 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return {"diff": float(a.mean()-b.mean()), "t": float(t), "p": float(p),
            "n_a": int(len(a)), "n_b": int(len(b))}

def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); i = 0; sa = a[order]
    while i < len(a):
        j2 = i
        while j2+1 < len(a) and sa[j2+1] == sa[i]: j2 += 1
        ranks[order[i:j2+1]] = (i+j2)/2.0+1; i = j2+1
    return ranks
def spear(x, y):
    rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else None

R = {}

# ---------- 收集 bp1 信号行(含微观测算) ----------
rows = []
for code, ft in feats.items():
    dates = ft["date"]; c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    h = ft["h"].astype(np.float64); l = ft["l"].astype(np.float64)
    n = len(c)
    for i in np.where(ft["bp1_sig"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if np.isnan(ft["t1_20"][i]): continue
        # 路径字段(入场=T+1开盘)
        mae_day = None
        if i+1 < n and o[i+1] > 0 and i+21 <= n:
            seg = l[i+1:i+21]
            if len(seg) == 20: mae_day = int(np.argmin(seg))
        f5 = b15 = None
        if i+21 <= n and o[i+1] > 0:
            f5 = (c[i+5]/o[i+1]-1.0)*PCT
            b15 = (c[i+20]/c[i+5]-1.0)*PCT
        gap = (o[i+1]/c[i]-1.0)*PCT if i+1 < n and c[i] > 0 else None
        intra1 = (c[i+1]/o[i+1]-1.0)*PCT if i+1 < n and o[i+1] > 0 else None
        rest19 = (c[i+20]/c[i+1]-1.0)*PCT if i+21 <= n and c[i+1] > 0 else None
        g = lambda k: (float(ft[k][i]) if not np.isnan(ft[k][i]) else None)
        rows.append({
            "code": code, "date": ds, "period": period_of(ds),
            "in_pool": 1 if code in pool_codes else 0,
            "n_raw": int(ft["n_raw"][i]),
            "d1": g("d1"), "d2": g("d2"), "d3": g("d3"), "d4": g("d4"), "d6": g("d6"),
            "rps": g("rps"), "l2_pass": bool(ft["l2_pass"][i]),
            "t1_20": float(ft["t1_20"][i])*PCT,
            "platform_range": g("platform_range"), "vol_atr_ratio": g("vol_atr_ratio"),
            "breakout_amp": g("breakout_amp"), "vr5_1": g("vr5_1"), "vr_long": g("vr_long"),
            "vp_corr": g("vp_corr"), "rsi": g("rsi"), "ma20_slope": g("ma20_slope"),
            "stage_mult": g("stage_mult"), "ext": g("ext"), "trend_age": g("trend_age"),
            "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
            "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
            "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
            "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
            "below_days": float(ft["below_days"][i]),
            "vol_ratio": g("vol_ratio"), "mae_day": mae_day,
            "gap": gap, "intra1": intra1, "rest19": rest19,
            "f5": f5, "b15": b15,
            "d0_intra": (c[i]/o[i]-1.0)*PCT if o[i] > 0 else None,   # 信号日T K线内(不入场, 参考)
        })
print("bp1信号(含微观测算): %d" % len(rows))

def layer_report(rs, key, bins, labels, split_period=True):
    out = {}
    for (lo, hi), lb in zip(bins, labels):
        sel = []
        for r in rs:
            v = r[key]
            if v is None or (isinstance(v, float) and np.isnan(v)): continue
            if lo is not None and v < lo: continue
            if hi is not None and v >= hi: continue
            sel.append(r)
        if not sel: continue
        d = desc([r["t1_20"] for r in sel])
        e = {"label": lb, "n": d["n"], "mean": d["mean"], "median": d["median"], "win": d["win"]}
        if split_period:
            for p in ["训练期", "验证期"]:
                sp = [r for r in sel if r["period"] == p]
                if sp: e[p] = desc([r["t1_20"] for r in sp])["mean"]
        out[lb] = e
    return out

def print_layers(title, v):
    print("  %s:" % title)
    for lb, e in v.items():
        print("    %-10s n=%6d mean=%+6.2f 中位=%+6.2f 胜%5.1f%% 训练=%s 验证=%s" % (
            lb, e["n"], e["mean"], e["median"], e["win"],
            ("%+.2f" % e["训练期"]) if "训练期" in e else "  -",
            ("%+.2f" % e["验证期"]) if "验证期" in e else "  -"))

# ---------- 模块1: 技术五条件质量分层(H1/H2) ----------
R["quality"] = {
    "platform_range": layer_report(rows, "platform_range", [(0,5),(5,10),(10,15),(15,20),(20,25)], ["<5%","5-10%","10-15%","15-20%","20-25%"]),
    "vol_atr_ratio": layer_report(rows, "vol_atr_ratio", [(0,0.02),(0.02,0.04),(0.04,0.05),(0.05,0.06)], ["<2%","2-4%","4-5%","5-6%"]),
    "breakout_amp": layer_report(rows, "breakout_amp", [(3,5),(5,8),(8,12),(12,None)], ["3-5%","5-8%","8-12%",">12%"]),
    "vr5_1": layer_report(rows, "vr5_1", [(1.5,2.0),(2.0,3.0),(3.0,5.0),(5.0,None)], ["1.5-2","2-3","3-5",">5"]),
    "vr_long": layer_report(rows, "vr_long", [(1.5,2.0),(2.0,3.0),(3.0,5.0),(5.0,None)], ["1.5-2","2-3","3-5",">5"]),
}
print("\n[模块1] 技术五条件质量分层(mean t1_20 %):")
for k, v in R["quality"].items(): print_layers(k, v)

# ---------- 模块2: N类评分分层(H4) ----------
R["nscore"] = {
    "n_raw": layer_report(rows, "n_raw", [(40,45),(45,50),(50,55),(55,60),(60,65),(65,None)], ["40-44","45-49","50-54","55-59","60-64",">=65"]),
    "d1_趋势结构": layer_report(rows, "d1", [(0,4),(4,8),(8,12),(12,16),(16,None)], ["<4","4-7","8-11","12-15",">15"]),
    "d2_量价关系": layer_report(rows, "d2", [(0,4),(4,8),(8,12),(12,16),(16,None)], ["<4","4-7","8-11","12-15",">15"]),
    "d3_相对强弱": layer_report(rows, "d3", [(0,4),(4,8),(8,12),(12,16),(16,None)], ["<4","4-7","8-11","12-15",">15"]),
    "d4_波动压缩": layer_report(rows, "d4", [(0,3),(3,6),(6,10),(10,13),(13,None)], ["<3","3-5","6-9","10-12",">12"]),
    "d6_筹码结构": layer_report(rows, "d6", [(0,7),(7,None)], ["<7",">=7"]),
}
print("\n[模块2] N类评分分层(mean t1_20 %):")
for k, v in R["nscore"].items(): print_layers(k, v)
# 各维度与t1_20的秩相关
for k in ["n_raw", "d1", "d2", "d3", "d4", "d6", "rps"]:
    vv = np.array([r[k] for r in rows if r[k] is not None], float)
    tt = np.array([r["t1_20"] for r in rows if r[k] is not None], float)
    m = ~np.isnan(vv)
    R["nscore"]["rankcorr_"+k] = spear(vv[m], tt[m])
print("  秩相关(n_raw/d1..d6/rps vs t1_20): " + " ".join(
    "%s=%s" % (k, "%+.3f" % R["nscore"]["rankcorr_"+k] if R["nscore"]["rankcorr_"+k] is not None else "N/A") for k in ["n_raw","d1","d2","d3","d4","d6","rps"]))

# ---------- 模块3: 对手方分层(H3) ----------
R["counterparty"] = {
    "pre20": layer_report(rows, "pre20", [(-200,0),(0,10),(10,20),(20,40),(40,None)], ["<0%","0-10%","10-20%","20-40%",">40%"]),
    "pre60": layer_report(rows, "pre60", [(-500,0),(0,20),(20,50),(50,100),(100,None)], ["<0%","0-20%","20-50%","50-100%",">100%"]),
    "dd20": layer_report(rows, "dd20", [(-100,-15),(-15,-10),(-10,-5),(-5,0)], ["<-15%","-15~-10%","-10~-5%",">-5%"]),
    "vol20": layer_report(rows, "vol20", [(0,2),(2,3),(3,4),(4,None)], ["<2%","2-3%","3-4%",">4%"]),
}
print("\n[模块3] 对手方分层(mean t1_20 %):")
for k, v in R["counterparty"].items(): print_layers(k, v)

# ---------- 模块4: 路径分析(H5) ----------
all_mae, all_mfe, all_f5, all_l15, all_below = [], [], [], [], []
for code, ft in feats.items():
    for i in range(119, len(ft["t1_20"])):
        if np.isnan(ft["t1_20"][i]) or np.isnan(ft["mae20"][i]): continue
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        all_mae.append(ft["mae20"][i]*PCT); all_mfe.append(ft["mfe20"][i]*PCT)
R["path"] = {
    "bp1信号": {"mae20": pdesc([r["mae20"] for r in rows]), "mfe20": pdesc([r["mfe20"] for r in rows]),
               "first5_min": pdesc([r["first5_min"] for r in rows]), "last15_max": pdesc([r["last15_max"] for r in rows]),
               "below_days": pdesc([r["below_days"] for r in rows])},
    "全A基准": {"mae20": pdesc(all_mae), "mfe20": pdesc(all_mfe)},
}
md = np.array([r["mae_day"] for r in rows if r["mae_day"] is not None], float)
R["path"]["bp1信号"]["mae_day_hist"] = {
    "前5日": float(100*(md < 5).mean()), "5-10日": float(100*((md >= 5) & (md < 10)).mean()),
    "10-20日": float(100*(md >= 10).mean()), "mean_day": float(md.mean()), "n": int(len(md))}
f5mae, l15mae = [], []
for code, ft in feats.items():
    dates = ft["date"]; l = ft["l"].astype(np.float64); o = ft["o"].astype(np.float64)
    n = len(l)
    for i in np.where(ft["bp1_sig"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if i+1 >= n or o[i+1] <= 0 or i+21 > n: continue
        e = o[i+1]
        f5mae.append((l[i+1:i+6].min()/e-1)*PCT)
        l15mae.append((l[i+6:i+21].min()/e-1)*PCT)
R["path"]["bp1信号"]["mae前5日"] = pdesc(f5mae)
R["path"]["bp1信号"]["mae后15日"] = pdesc(l15mae)
print("\n[模块4] 路径分析:")
for grp in ["bp1信号", "全A基准"]:
    p = R["path"][grp]
    print("  %-6s MAE med=%+6.2f%% mean=%+6.2f%% | MFE med=%+6.2f%% mean=%+6.2f%%" % (
        grp, p["mae20"]["median"], p["mae20"]["mean"], p["mfe20"]["median"], p["mfe20"]["mean"]))
bp = R["path"]["bp1信号"]
print("  bp1: 前5日最低%+.2f%% / 后15日最低%+.2f%% / below成本天数中位%.0f / 首5日最低中位%+.2f%% / 后15日最高中位%+.2f%%" % (
    bp["mae前5日"]["median"], bp["mae后15日"]["median"], bp["below_days"]["median"],
    bp["first5_min"]["median"], bp["last15_max"]["median"]))
print("  bp1 MAE发生时点: 前5日%.1f%% 5-10日%.1f%% 10-20日%.1f%% (平均第%.1f日)" % (
    bp["mae_day_hist"]["前5日"], bp["mae_day_hist"]["5-10日"], bp["mae_day_hist"]["10-20日"], bp["mae_day_hist"]["mean_day"]))

# ---------- 模块5: 量能事前性(H2) ----------
vr5q = np.percentile([r["vr5_1"] for r in rows if r["vr5_1"] is not None], [0, 20, 40, 60, 80, 100])
R["vr5_trade"] = {}
for qi in range(5):
    lo, hi = vr5q[qi], vr5q[qi+1]
    sel = [r for r in rows if r["vr5_1"] is not None and r["f5"] is not None and r["b15"] is not None
           and (r["vr5_1"] >= lo if qi == 4 else lo <= r["vr5_1"] < hi)]
    if not sel: continue
    R["vr5_trade"]["Q%d(%.1f-%.1f)" % (qi+1, lo, hi)] = {
        "n": len(sel), "前5日": desc([r["f5"] for r in sel])["mean"], "后15日": desc([r["b15"] for r in sel])["mean"],
        "合计": desc([r["t1_20"] for r in sel])["mean"],
        "前5日_胜率": desc([r["f5"] for r in sel])["win"], "后15日_胜率": desc([r["b15"] for r in sel])["win"]}
vr = np.array([r["vr5_1"] for r in rows]); f5v = np.array([r["f5"] for r in rows]); b15v = np.array([r["b15"] for r in rows])
m = ~np.isnan(f5v) & ~np.isnan(b15v) & ~np.isnan(vr)
R["vr5_trade"]["秩相关"] = {"vr5_vs_前5日": spear(vr[m], f5v[m]), "vr5_vs_后15日": spear(vr[m], b15v[m]), "n": int(m.sum())}
R["volratio"] = layer_report(rows, "vol_ratio", [(0,0.8),(0.8,1.2),(1.2,1.6),(1.6,None)], ["<0.8","0.8-1.2","1.2-1.6",">1.6"], split_period=False)
print("\n[模块5] 量能事前性(vr5_1五分位):")
for k, v in R["vr5_trade"].items():
    if k == "秩相关":
        print("  秩相关: vr5_1 vs 前5日 %+.3f | vs 后15日 %+.3f (n=%d)" % (v["vr5_vs_前5日"], v["vr5_vs_后15日"], v["n"]))
    else:
        print("  %-14s n=%5d 前5日%+6.2f%%(胜%.0f%%) 后15日%+6.2f%%(胜%.0f%%) 合计%+6.2f%%" % (
            k, v["n"], v["前5日"], v["前5日_胜率"], v["后15日"], v["后15日_胜率"], v["合计"]))
print("  vol_ratio(事后跟风量, T+1~T+20均量/T-20~T均量)分层:")
for lb, e in R["volratio"].items():
    print("    %-8s n=%6d mean=%+6.2f%%" % (lb, e["n"], e["mean"]))

# ---------- 模块6: 入场滑点 + 对数分解 ----------
gp = [r["gap"] for r in rows if r["gap"] is not None]
R["gap"] = pdesc(gp)
R["gap"]["高开>1%占比"] = float(100*np.mean(np.array(gp) > 1.0))
R["gap"]["低开>1%占比"] = float(100*np.mean(np.array(gp) < -1.0))
R["gap"]["gap_vs_t1_20_秩相关"] = spear(np.array(gp), np.array([r["t1_20"] for r in rows if r["gap"] is not None]))
# 对数分解: log(1+t1_20) = log(o[T+1]/c[T]) + log(c[T+1]/o[T+1]) + log(c[T+20]/c[T+1])
lg = np.array([r["gap"] for r in rows if r["gap"] is not None and r["intra1"] is not None and r["rest19"] is not None])
li = np.array([r["intra1"] for r in rows if r["gap"] is not None and r["intra1"] is not None and r["rest19"] is not None])
lr = np.array([r["rest19"] for r in rows if r["gap"] is not None and r["intra1"] is not None and r["rest19"] is not None])
lt = np.array([r["t1_20"] for r in rows if r["gap"] is not None and r["intra1"] is not None and r["rest19"] is not None])
with np.errstate(invalid="ignore", divide="ignore"):
    L = np.log(1+lt/100.0); Lg = np.log(1+lg/100.0); Li = np.log(1+li/100.0); Lr = np.log(1+lr/100.0)
ok = ~np.isnan(L) & ~np.isnan(Lg) & ~np.isnan(Li) & ~np.isnan(Lr)
R["decompose"] = {
    "总log收益": float(L[ok].sum()),
    "T收->T+1开(跳空)": float(Lg[ok].sum()),
    "T+1开->T+1收(首日)": float(Li[ok].sum()),
    "T+1收->T+21收(余19日)": float(Lr[ok].sum()),
    "n": int(ok.sum()),
    "校验_加总": float((Lg[ok]+Li[ok]+Lr[ok]).sum()),
}
print("\n[模块6] 入场滑点(T+1开盘跳空): mean=%+.2f%% median=%+.2f%% p10=%+.2f%% p90=%+.2f%% 高开>1%%占比%.1f%% 低开>1%%占比%.1f%%" % (
    R["gap"]["mean"], R["gap"]["median"], R["gap"]["p10"], R["gap"]["p90"],
    R["gap"]["高开>1%占比"], R["gap"]["低开>1%占比"]))
print("  gap vs t1_20 秩相关: %+.3f" % R["gap"]["gap_vs_t1_20_秩相关"])
print("  收益对数分解(占总log收益比例): 跳空%+.1f%% 首日%+.1f%% 余19日%+.1f%% (总%+.1f, 校验%+.1f, n=%d)" % (
    100*R["decompose"]["T收->T+1开(跳空)"]/R["decompose"]["总log收益"],
    100*R["decompose"]["T+1开->T+1收(首日)"]/R["decompose"]["总log收益"],
    100*R["decompose"]["T+1收->T+21收(余19日)"]/R["decompose"]["总log收益"],
    R["decompose"]["总log收益"], R["decompose"]["校验_加总"], R["decompose"]["n"]))

json.dump(R, open(os.path.join(HERE, "bp1_micro_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> bp1_micro_results.json")
