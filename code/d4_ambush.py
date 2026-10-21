# -*- coding: utf-8 -*-
"""
d4 波动率压缩独立因子 — A级修复验证：突破前埋伏 vs 突破后追入
信号定义（全部 ~bp1_tech，即未触发技术五条件 = 未突破状态）:
  amb_loose : d4>=10 & plat<25 & ~tech
  amb_mid   : d4>=12 & plat<25 & ~tech
  amb_tight : d4>=15 & plat<25 & ~tech
  amb_full  : d4>=12 & plat<25 & vatr<0.06 & ~tech
  d4_only   : d4>=12
  d4_plat   : d4>=12 & plat<25
  fresh     : d4>=12 & prev_d4<12 & plat<25 & ~tech  (今日刚进入压缩态)
对照: bp1_sig（突破后追入）、全宇宙基准
口径: T+1开盘入场 → T+20收盘出场 (t1_20)，训练/验证切分，池内外
"""
import pickle, os, json
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
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float(100*(v > 0).mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "std": float(v.std())}

# ---------- 预扫: 全宇宙基准 ----------
base_all, base_tr, base_va = [], [], []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    for i in range(len(dates)):
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        v = float(t1[i])*PCT
        base_all.append(v)
        (base_tr if ds < TRAIN_END else base_va).append(v)

R = {"baseline": {"全部": desc(base_all), "训练期": desc(base_tr), "验证期": desc(base_va)}}

# ---------- 信号掩码收集 ----------
def collect(mask_fn):
    """返回 dict[(period, pool_flag)] -> list of t1_20"""
    out = {}
    for code, ft in feats.items():
        dates = ft["date"]; t1 = ft["t1_20"]
        m = mask_fn(ft)
        if m is None or not m.any(): continue
        in_pool = code in pool_codes
        for i in np.where(m)[0]:
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            key = (period_of(ds), in_pool)
            out.setdefault(key, []).append(float(t1[i])*PCT)
    return out

def summarize(out):
    res = {}
    for grp_name, sel in [("全部", lambda p, ip: True), ("训练期", lambda p, ip: p == "训练期"),
                          ("验证期", lambda p, ip: p == "验证期"),
                          ("池内", lambda p, ip: ip), ("池外", lambda p, ip: not ip),
                          ("池内-训练", lambda p, ip: ip and p == "训练期"),
                          ("池内-验证", lambda p, ip: ip and p == "验证期"),
                          ("池外-训练", lambda p, ip: (not ip) and p == "训练期"),
                          ("池外-验证", lambda p, ip: (not ip) and p == "验证期")]:
        vals = []
        for (p, ip), lst in out.items():
            if sel(p, ip): vals += lst
        res[grp_name] = desc(vals)
    return res

def m_plat(ft): return ft["platform_range"]
def m_d4(ft): return ft["d4"]
def m_tech(ft): return ft["bp1_tech"] > 0 if ft["bp1_tech"].dtype != bool else ft["bp1_tech"]
def m_vatr(ft): return ft["vol_atr_ratio"]

defs = {
    "amb_loose(d4>=10,plat<25,未突破)": lambda ft: (m_d4(ft) >= 10) & (m_plat(ft) < 25) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft)),
    "amb_mid(d4>=12,plat<25,未突破)": lambda ft: (m_d4(ft) >= 12) & (m_plat(ft) < 25) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft)),
    "amb_tight(d4>=15,plat<25,未突破)": lambda ft: (m_d4(ft) >= 15) & (m_plat(ft) < 25) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft)),
    "amb_full(d4>=12,plat<25,vatr<6%,未突破)": lambda ft: (m_d4(ft) >= 12) & (m_plat(ft) < 25) & (m_vatr(ft) < 6) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft)) & ~np.isnan(m_vatr(ft)),
    "d4_only(d4>=12)": lambda ft: (m_d4(ft) >= 12) & ~np.isnan(m_d4(ft)),
    "d4_plat(d4>=12,plat<25)": lambda ft: (m_d4(ft) >= 12) & (m_plat(ft) < 25) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft)),
    "bp1_sig(突破后追入,对照)": lambda ft: ft["bp1_sig"] > 0 if ft["bp1_sig"].dtype != bool else ft["bp1_sig"],
}

for name, fn in defs.items():
    out = collect(fn)
    R[name] = summarize(out)
    n = R[name]["全部"]["n"] if R[name]["全部"] else 0
    m = R[name]["全部"]["mean"] if R[name]["全部"] else float("nan")
    mv = R[name]["验证期"]["mean"] if R[name]["验证期"] else float("nan")
    print("%-42s n=%7d mean=%+.2f%% valid=%+.2f%%" % (name, n, m, mv))

# ---------- fresh 信号: 今日刚进入压缩态 ----------
def fresh_fn(ft):
    d4 = m_d4(ft); plat = m_plat(ft); tech = m_tech(ft)
    ok = (d4 >= 12) & (plat < 25) & (~tech) & ~np.isnan(d4) & ~np.isnan(plat)
    prev_ok = np.zeros_like(ok)
    prev_ok[1:] = (d4[:-1] < 12) | np.isnan(d4[:-1])
    return ok & prev_ok

out = collect(fresh_fn)
R["fresh(今日d4上穿12,未突破)"] = summarize(out)
print("%-42s n=%7d mean=%+.2f%%" % ("fresh", R["fresh(今日d4上穿12,未突破)"]["全部"]["n"],
                                     R["fresh(今日d4上穿12,未突破)"]["全部"]["mean"]))

# ---------- amb_mid 十分位 ----------
vals = []
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    m = (m_d4(ft) >= 12) & (m_plat(ft) < 25) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft))
    for i in np.where(m)[0]:
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        vals.append((float(t1[i])*PCT, float(m_d4(ft)[i])))
vals.sort(key=lambda x: x[0])
n = len(vals); dec = {}
for k in range(10):
    seg = vals[int(n*k/10):int(n*(k+1)/10)]
    if not seg: continue
    arr = np.array([s[0] for s in seg])
    dec["D%d" % (k+1)] = {"n": len(seg), "mean": float(arr.mean()), "median": float(np.median(arr)),
                          "win": float(100*(arr > 0).mean()),
                          "range": "%.1f~%.1f" % (seg[0][0], seg[-1][0])}
R["amb_mid_deciles"] = dec

# ---------- 月度分布 amb_mid ----------
from collections import defaultdict
mon = defaultdict(list)
for code, ft in feats.items():
    dates = ft["date"]; t1 = ft["t1_20"]
    m = (m_d4(ft) >= 12) & (m_plat(ft) < 25) & (~m_tech(ft)) & ~np.isnan(m_d4(ft)) & ~np.isnan(m_plat(ft))
    for i in np.where(m)[0]:
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        mon[ds[:6]].append(float(t1[i])*PCT)
R["amb_monthly"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(mon.items())}

json.dump(R, open(os.path.join(HERE, "d4_ambush_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved -> d4_ambush_results.json")
