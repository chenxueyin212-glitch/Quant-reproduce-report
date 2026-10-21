# -*- coding: utf-8 -*-
"""
B2因子重构评估 — Step5 微观假设检验
B2机制: 候选日T(J<20+触知行多空线回收+b1_score>=5) -> 确认日T+1放量阳线(b2_score>=5)
        -> T+2开盘入场 -> 持有20个交易日 -> T+21收盘出场

B2微观经济学假设(对应用户框架):
  H1(趋势延续/确认有效): 放量阳线=新资金确认, 趋势未坏, 确认后应继续跑赢
  H2(赚谁的钱): 确认日被阳线"吓下车"的人 + 随后跟风上车者的钱
  H3(被谁赚钱): 候选日附近埋伏更早者(确认日出货) / 认为"确认=动能耗尽"者的钱

检验模块:
  1. 确认日质量分层(vr5/chg/上影线/J/score) — H1直接检验: 质量刻度有无判别力
  2. 确认机制增量分解 — 配对归因(A/B)分期 + 确认日baked-in分解 + 显著性
  3. 对手方分层(pre20/pre60/dd20/cand_score) — H2/H3
  4. 路径分析(B2 vs B1候选 vs 基准; MAE发生时点) — H1/H3
  5. vr5事前可交易性复测(前5日/后15日分解 + vol_ratio) — H2
  6. 入场滑点(T+2开盘跳空)
"""
import pickle, os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d2 = pickle.load(open(os.path.join(HERE, "b2_features.pkl"), "rb"))
feats = d2["feats"]
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

def welch(a, b):
    """Welch t检验, 返回 (差, t, p双侧近似)"""
    a = np.array(a, float); b = np.array(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 30 or len(b) < 30: return None
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = math.sqrt(va+vb)
    if se == 0: return None
    t = (a.mean()-b.mean())/se
    # 正态近似p值(双侧)
    p = 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return {"diff": float(a.mean()-b.mean()), "t": float(t), "p": float(p),
            "n_a": int(len(a)), "n_b": int(len(b))}

R = {}

# ---------- 收集 B2 信号行(含路径/对手方/滑点字段) ----------
rows = []
for code, ft in feats.items():
    dates = ft["date"]; c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    h = ft["h"].astype(np.float64); l = ft["l"].astype(np.float64)
    n = len(c)
    for i in np.where(ft["b2_sig"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if np.isnan(ft["t1_20"][i]): continue
        di = i-1  # 候选日
        # MAE发生时点(基于T+2开盘入场窗口)
        mae_day = None
        if i+1 < n and o[i+1] > 0 and i+21 <= n:
            seg = l[i+1:min(i+21, n)]
            if len(seg) == 20: mae_day = int(np.argmin(seg))
        # 前5/后15分解(入场=T+2开盘)
        f5 = b15 = None
        if i+21 <= n and o[i+1] > 0:
            f5 = c[i+5]/o[i+1]-1.0
            b15 = c[i+20]/c[i+5]-1.0
        rows.append({
            "code": code, "date": ds, "period": period_of(ds),
            "in_pool": 1 if code in pool_codes else 0,
            "strong": bool(ft["b2_strong"][i]),
            "t1_20": float(ft["t1_20"][i])*PCT,
            "vr5": float(ft["vr5"][i]), "chg": float(ft["chg"][i]),
            "upper_shadow": float(ft["upper_shadow"][i]), "j": float(ft["j"][i]),
            "b2_score": float(ft["b2_score"][i]),
            "cand_score": float(ft["b1_cand_score"][di]) if di >= 0 and not np.isnan(ft["b1_cand_score"][di]) else np.nan,
            "pre20": float(ft["pre20"][i])*PCT, "pre60": float(ft["pre60"][i])*PCT,
            "dd20": float(ft["dd20"][i])*PCT, "vol20": float(ft["vol20"][i])*PCT,
            "mae20": float(ft["mae20"][i])*PCT, "mfe20": float(ft["mfe20"][i])*PCT,
            "first5_min": float(ft["first5_min"][i])*PCT, "last15_max": float(ft["last15_max"][i])*PCT,
            "below_days": float(ft["below_days"][i]),
            "vol_ratio": float(ft["vol_ratio"][i]),
            "mae_day": mae_day,
            "d0_intra": float(c[i]/o[i]-1)*PCT,        # 确认日 T+1 开盘->收盘
            "d0_full": float(c[i]/c[di]-1)*PCT if di >= 0 else None,  # T收->T+1收
            "ex_d0": float(c[i+20]/c[i]-1)*PCT if i+20 < n else None,  # T+1收->T+21收(剔除确认日)
            "gap": float(o[i+1]/c[i]-1)*PCT if i+1 < n else None,      # T+1收->T+2开盘跳空
            "f5": f5*PCT if f5 is not None else None,
            "b15": b15*PCT if b15 is not None else None,
        })
print("B2信号(含微观测算): %d" % len(rows))

def layer_report(rs, key, bins, labels, split_period=True):
    """按 key 分层统计 t1_20; bins=[(lo,hi),...] lo/hi 为 None 表示开区间"""
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

# ---------- 模块1: 确认日质量分层(H1) ----------
R["quality"] = {
    "vr5": layer_report(rows, "vr5", [(0.0,1.2),(1.2,2.0),(2.0,3.0),(3.0,None)], ["<1.2","1.2-2.0","2.0-3.0",">3.0"]),
    "chg": layer_report(rows, "chg", [(0.0,1.5),(1.5,3.9),(3.9,7.0),(7.0,None)], ["0-1.5%","1.5-3.9%","3.9-7%",">7%"]),
    "upper_shadow": layer_report(rows, "upper_shadow", [(0.0,0.5),(0.5,1.0),(1.0,2.0),(2.0,None)], ["<0.5","0.5-1.0","1.0-2.0",">2.0"]),
    "j_confirm": layer_report(rows, "j", [(0.0,20.0),(20.0,40.0),(40.0,55.0),(55.0,None)], ["<20","20-40","40-55",">55"]),
    "b2_score": layer_report(rows, "b2_score", [(5.0,6.0),(6.0,7.0),(7.0,8.0),(8.0,None)], ["5","6","7",">=8"]),
}
print("\n[模块1] 确认日质量分层(mean t1_20 %):")
for k, v in R["quality"].items():
    print("  %s:" % k)
    for lb, e in v.items():
        print("    %-8s n=%6d mean=%+6.2f 训练=%s 验证=%s" % (
            lb, e["n"], e["mean"],
            ("%+.2f" % e["训练期"]) if "训练期" in e else "  -",
            ("%+.2f" % e["验证期"]) if "验证期" in e else "  -"))

# ---------- 模块2: 确认机制增量分解 ----------
# A/B配对(同一事件): A=候选日T+1开盘入场 t1_20[di]; B=确认后T+2开盘入场 t1_20[dn]
pairA, pairB, pairPeriod = [], [], []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["b1_cand"])[0]:
        if i < 1 or i+1 >= len(ft["t1_20"]): continue
        ds_c = dates[i].replace("-", "")
        if not (D0 <= ds_c <= D1): continue
        a = ft["t1_20"][i]; b = ft["t1_20"][i+1]
        if np.isnan(a) or np.isnan(b): continue
        pairA.append(float(a)*PCT); pairB.append(float(b)*PCT)
        pairPeriod.append(period_of(ds_c))
pairA = np.array(pairA); pairB = np.array(pairB)
R["paired"] = {"A_B1候选直接进": desc(pairA), "B_B2确认后进": desc(pairB)}
R["paired"]["A_minus_B"] = desc(pairA-pairB)
for p in ["训练期", "验证期"]:
    m = np.array([q == p for q in pairPeriod])
    R["paired"]["A_" + p] = desc(pairA[m]); R["paired"]["B_" + p] = desc(pairB[m])
    R["paired"]["AminusB_" + p] = desc(pairA[m]-pairB[m])
print("\n[模块2a] 配对归因(同事件, 等一天再进):")
print("  A(候选直接T+1开): mean=%+.2f%%  B(确认后T+2开): mean=%+.2f%%  差=%+.2f%% (n=%d)" % (
    pairA.mean(), pairB.mean(), pairA.mean()-pairB.mean(), len(pairA)))
for p in ["训练期", "验证期"]:
    m = np.array([q == p for q in pairPeriod])
    print("  %s: A=%+.2f%% B=%+.2f%% 差=%+.2f%% (n=%d)" % (p, pairA[m].mean(), pairB[m].mean(), pairA[m].mean()-pairB[m].mean(), m.sum()))

# 确认日baked-in分解: 被确认候选的 t1_20[di] = d0_intra(确认日K线内) + ex_d0(T+1收->T+21收) + gap效应
conf_rows = [r for r in rows if not (r["cand_score"] is None or (isinstance(r["cand_score"], float) and np.isnan(r["cand_score"])))]
# 直接用B2信号行: t1_20[dn] 已是T+2入场; 需要候选日入场视角 -> 重算
conf_t1di, conf_d0intra, conf_exd0, conf_t1dn = [], [], [], []
unconf_t1di, unconf_d0intra, unconf_exd0, unconf_t2 = [], [], [], []
unconf_period, conf_period = [], []
for code, ft in feats.items():
    dates = ft["date"]; c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    n = len(c)
    for di in np.where(ft["b1_cand"])[0]:
        ds = dates[di].replace("-", "")
        if not (D0 <= ds <= D1): continue
        dn = di+1
        if di+1 >= n or di+21 >= n: continue
        t1di = ft["t1_20"][di]          # 候选日T+1开盘入场
        if np.isnan(t1di): continue
        confirmed = bool(ft["b2_sig"][dn])
        d0i = c[dn]/o[dn]-1.0           # 确认日/非确认日 T+1开盘->收盘
        exd0 = c[dn+20]/c[dn]-1.0       # T+1收->T+21收
        if confirmed:
            conf_t1di.append(t1di*PCT); conf_d0intra.append(d0i*PCT)
            conf_exd0.append(exd0*PCT); conf_t1dn.append(ft["t1_20"][dn]*PCT); conf_period.append(period_of(ds))
        else:
            unconf_t1di.append(t1di*PCT); unconf_d0intra.append(d0i*PCT)
            unconf_exd0.append(exd0*PCT); unconf_t2.append(float(c[dn+20]/o[dn+1]-1)*PCT)
            unconf_period.append(period_of(ds))
R["decompose"] = {
    "被确认_候选日入场": desc(conf_t1di),
    "被确认_确认日K线内(d0_intra)": desc(conf_d0intra),
    "被确认_剔除确认日(T+1收->T+21收)": desc(conf_exd0),
    "被确认_B2口径(T+2入场)": desc(conf_t1dn),
    "未被确认_候选日入场": desc(unconf_t1di),
    "未被确认_非确认日K线内(d0_intra)": desc(unconf_d0intra),
    "未被确认_剔除确认日(T+1收->T+21收)": desc(unconf_exd0),
    "未被确认_同T+2入场": desc(unconf_t2),
}
R["decompose"]["增量检验_被确认T2入场_vs_未确认T2入场"] = welch(conf_t1dn, unconf_t2)
for p in ["训练期", "验证期"]:
    mc = np.array([q == p for q in conf_period]); mu = np.array([q == p for q in unconf_period])
    R["decompose"]["分期_" + p] = {
        "被确认_T2入场": desc(np.array(conf_t1dn)[mc])["mean"] if mc.sum() else None,
        "未确认_T2入场": desc(np.array(unconf_t2)[mu])["mean"] if mu.sum() else None,
    }
print("\n[模块2b] 确认日baked-in分解(候选集内):")
for k in ["被确认_候选日入场", "被确认_确认日K线内(d0_intra)", "被确认_剔除确认日(T+1收->T+21收)",
          "被确认_B2口径(T+2入场)", "未被确认_候选日入场", "未被确认_非确认日K线内(d0_intra)",
          "未被确认_剔除确认日(T+1收->T+21收)", "未被确认_同T+2入场"]:
    v = R["decompose"][k]
    print("  %-32s n=%6d mean=%+7.2f%% median=%+7.2f%%" % (k, v["n"], v["mean"], v["median"]))
w = R["decompose"]["增量检验_被确认T2入场_vs_未确认T2入场"]
print("  确认机制真实增量(同T+2入场): %+.2f pct  t=%.2f p=%.4f (n=%d/%d)" % (
    w["diff"], w["t"], w["p"], w["n_a"], w["n_b"]))
for p in ["训练期", "验证期"]:
    print("  %s: 被确认T2=%+.2f%%  未确认T2=%+.2f%%" % (p, R["decompose"]["分期_"+p]["被确认_T2入场"], R["decompose"]["分期_"+p]["未确认_T2入场"]))

# ---------- 模块3: 对手方分层(H2/H3) ----------
R["counterparty"] = {
    "pre20": layer_report(rows, "pre20", [(-200.0,0.0),(0.0,20.0),(20.0,40.0),(40.0,80.0),(80.0,None)], ["<0%","0-20%","20-40%","40-80%",">80%"]),
    "pre60": layer_report(rows, "pre60", [(-500.0,0.0),(0.0,20.0),(20.0,50.0),(50.0,100.0),(100.0,None)], ["<0%","0-20%","20-50%","50-100%",">100%"]),
    "dd20": layer_report(rows, "dd20", [(-100.0,-15.0),(-15.0,-10.0),(-10.0,-5.0),(-5.0,0.0)], ["<-15%","-15~-10%","-10~-5%",">-5%"]),
    "cand_score": layer_report(rows, "cand_score", [(5.0,6.0),(6.0,7.0),(7.0,8.0),(8.0,None)], ["5","6","7",">=8"]),
}
print("\n[模块3] 对手方分层(mean t1_20 %):")
for k, v in R["counterparty"].items():
    print("  %s:" % k)
    for lb, e in v.items():
        print("    %-8s n=%6d mean=%+6.2f 训练=%s 验证=%s" % (
            lb, e["n"], e["mean"],
            ("%+.2f" % e["训练期"]) if "训练期" in e else "  -",
            ("%+.2f" % e["验证期"]) if "验证期" in e else "  -"))

# ---------- 模块4: 路径分析 ----------
# B2信号 vs B1候选(全部) vs 全A基准
b1_mae, b1_mfe, b1_f5, b1_l15, b1_below = [], [], [], [], []
all_mae, all_mfe = [], []
for code, ft in feats.items():
    dates = ft["date"]
    for i in np.where(ft["b1_cand"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if np.isnan(ft["mae20"][i]): continue
        b1_mae.append(ft["mae20"][i]*PCT); b1_mfe.append(ft["mfe20"][i]*PCT)
        b1_f5.append(ft["first5_min"][i]*PCT); b1_l15.append(ft["last15_max"][i]*PCT)
        b1_below.append(ft["below_days"][i])
    for i in range(119, len(ft["t1_20"])):
        if np.isnan(ft["t1_20"][i]) or np.isnan(ft["mae20"][i]): continue
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        all_mae.append(ft["mae20"][i]*PCT); all_mfe.append(ft["mfe20"][i]*PCT)
def pdesc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
R["path"] = {
    "B2信号": {"mae20": pdesc([r["mae20"] for r in rows]), "mfe20": pdesc([r["mfe20"] for r in rows]),
               "first5_min": pdesc([r["first5_min"] for r in rows]), "last15_max": pdesc([r["last15_max"] for r in rows]),
               "below_days": pdesc([r["below_days"] for r in rows])},
    "B1候选": {"mae20": pdesc(b1_mae), "mfe20": pdesc(b1_mfe), "first5_min": pdesc(b1_f5),
               "last15_max": pdesc(b1_l15), "below_days": pdesc(b1_below)},
    "全A基准": {"mae20": pdesc(all_mae), "mfe20": pdesc(all_mfe)},
}
# MAE发生时点分布(B2)
md = [r["mae_day"] for r in rows if r["mae_day"] is not None]
md = np.array(md, float)
R["path"]["B2信号"]["mae_day_hist"] = {
    "前5日": float(100*(md < 5).mean()), "5-10日": float(100*((md >= 5) & (md < 10)).mean()),
    "10-20日": float(100*(md >= 10).mean()), "mean_day": float(md.mean()), "n": int(len(md))}
# MAE分时段: 前5日MAE vs 后15日MAE(新算)
f5mae, l15mae = [], []
for code, ft in feats.items():
    dates = ft["date"]; l = ft["l"].astype(np.float64); o = ft["o"].astype(np.float64)
    n = len(l)
    for i in np.where(ft["b2_sig"])[0]:
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if i+1 >= n or o[i+1] <= 0 or i+21 > n: continue
        e = o[i+1]
        f5mae.append((l[i+1:i+6].min()/e-1)*PCT)
        l15mae.append((l[i+6:i+21].min()/e-1)*PCT)
R["path"]["B2信号"]["mae前5日"] = pdesc(f5mae)
R["path"]["B2信号"]["mae后15日"] = pdesc(l15mae)
print("\n[模块4] 路径分析:")
for grp in ["B2信号", "B1候选", "全A基准"]:
    p = R["path"][grp]
    print("  %-6s MAE med=%+6.2f%% mean=%+6.2f%% | MFE med=%+6.2f%% mean=%+6.2f%%" % (
        grp, p["mae20"]["median"], p["mae20"]["mean"], p["mfe20"]["median"], p["mfe20"]["mean"]))
b2p = R["path"]["B2信号"]
print("  B2: 前5日最低%+.2f%% / 后15日最低%+.2f%% / 持仓below成本天数中位%.0f" % (
    b2p["first5_min"]["median"], b2p["mae后15日"]["median"], b2p["below_days"]["median"]))
print("  B2 MAE发生时点: 前5日%.1f%% 5-10日%.1f%% 10-20日%.1f%% (平均第%.1f日)" % (
    b2p["mae_day_hist"]["前5日"], b2p["mae_day_hist"]["5-10日"], b2p["mae_day_hist"]["10-20日"], b2p["mae_day_hist"]["mean_day"]))

# ---------- 模块5: vr5事前可交易性复测(H2) ----------
# 确认日vr5五分位 -> 前5日(T+2~T+6) / 后15日(T+7~T+21) 收益
vr5q = np.percentile([r["vr5"] for r in rows], [0, 20, 40, 60, 80, 100])
R["vr5_trade"] = {}
for qi in range(5):
    lo, hi = vr5q[qi], vr5q[qi+1]
    sel = [r for r in rows if r["vr5"] is not None and (r["vr5"] >= lo if qi == 4 else lo <= r["vr5"] < hi) and r["f5"] is not None and r["b15"] is not None]
    if not sel: continue
    R["vr5_trade"]["Q%d(%.1f-%.1f)" % (qi+1, lo, hi)] = {
        "n": len(sel), "前5日": desc([r["f5"] for r in sel])["mean"], "后15日": desc([r["b15"] for r in sel])["mean"],
        "合计": desc([r["t1_20"] for r in sel])["mean"],
        "前5日_胜率": desc([r["f5"] for r in sel])["win"], "后15日_胜率": desc([r["b15"] for r in sel])["win"]}
# vr5 vs 后15日秩相关
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
vr = np.array([r["vr5"] for r in rows]); f5v = np.array([r["f5"] for r in rows]); b15v = np.array([r["b15"] for r in rows])
m = ~np.isnan(f5v) & ~np.isnan(b15v)
R["vr5_trade"]["秩相关"] = {"vr5_vs_前5日": spear(vr[m], f5v[m]), "vr5_vs_后15日": spear(vr[m], b15v[m]), "n": int(m.sum())}
# vol_ratio(事后跟风量)分层
R["volratio"] = layer_report(rows, "vol_ratio", [(0.0,0.8),(0.8,1.2),(1.2,1.6),(1.6,None)], ["<0.8","0.8-1.2","1.2-1.6",">1.6"], split_period=False)
print("\n[模块5] vr5事前可交易性:")
for k, v in R["vr5_trade"].items():
    if k == "秩相关":
        print("  秩相关: vr5 vs 前5日 %+.3f | vr5 vs 后15日 %+.3f (n=%d)" % (v["vr5_vs_前5日"], v["vr5_vs_后15日"], v["n"]))
    else:
        print("  %-14s n=%5d 前5日%+6.2f%%(胜%.0f%%) 后15日%+6.2f%%(胜%.0f%%) 合计%+6.2f%%" % (
            k, v["n"], v["前5日"], v["前5日_胜率"], v["后15日"], v["后15日_胜率"], v["合计"]))
print("  vol_ratio(事后跟风量)分层:")
for lb, e in R["volratio"].items():
    print("    %-8s n=%6d mean=%+6.2f%%" % (lb, e["n"], e["mean"]))

# ---------- 模块6: 入场滑点 ----------
gp = [r["gap"] for r in rows if r["gap"] is not None]
R["gap"] = pdesc(gp)
R["gap"]["高开>1%占比"] = float(100*np.mean(np.array(gp) > 1.0))
R["gap"]["低开>1%占比"] = float(100*np.mean(np.array(gp) < -1.0))
R["gap"]["gap_vs_t1_20_秩相关"] = spear(np.array(gp), np.array([r["t1_20"] for r in rows if r["gap"] is not None]))
print("\n[模块6] T+2开盘跳空(入场滑点): mean=%+.2f%% median=%+.2f%% p10=%+.2f%% p90=%+.2f%% 高开>1%%占比%.1f%% 低开>1%%占比%.1f%%" % (
    R["gap"]["mean"], R["gap"]["median"], R["gap"]["p10"], R["gap"]["p90"],
    R["gap"]["高开>1%占比"], R["gap"]["低开>1%占比"]))
print("  gap vs t1_20 秩相关: %+.3f" % R["gap"]["gap_vs_t1_20_秩相关"])

json.dump(R, open(os.path.join(HERE, "b2_micro_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> b2_micro_results.json")
