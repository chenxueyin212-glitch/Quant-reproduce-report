# -*- coding: utf-8 -*-
"""B2因子重构评估 — Step4 稳健性检验(OFAT + regime + 截断)"""
import pickle, os, json, math, statistics as st, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0

d = pickle.load(open(os.path.join(HERE, "b2_features.pkl"), "rb"))
feats = d["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
codes = list(feats.keys())

def ma_cumsum(arr, p):
    n = len(arr); r = np.full(n, np.nan)
    if n < p: return r
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    r[p-1:] = (cs[period:] if False else (cs[p:] - cs[:-p]))/p
    return r

def trend_system_A(l, h, it, ip, di):
    lo = max(0, di-56); hi = di-2
    if hi-lo < 2: return False
    tr = l[lo:hi][it[lo:hi]]; pk = h[lo:hi][ip[lo:hi]]
    return len(tr) >= 2 and len(pk) >= 2 and tr[-1] > tr[-2] and pk[-1] > pk[-2]

def recompute(ft, P):
    """按参数P重算 b1_score/b2_score/信号"""
    c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    h = ft["h"].astype(np.float64); l = ft["l"].astype(np.float64)
    J = ft["j"].astype(np.float64); zt = ft["zt"].astype(np.float64); zd = ft["zd"].astype(np.float64)
    ma5, ma13, ma20 = ft["ma5"].astype(np.float64), ft["ma13"].astype(np.float64), ft["ma20"].astype(np.float64)
    ma40, ma60 = ft["ma40"].astype(np.float64), ft["ma60"].astype(np.float64)
    dif, dea = ft["dif"].astype(np.float64), ft["dea"].astype(np.float64)
    vr5, chg = ft["vr5"].astype(np.float64), ft["chg"].astype(np.float64)
    bull = ft["bullish"]; us = ft["upper_shadow"].astype(np.float64)
    it, ip = ft["is_trough"], ft["is_peak"]
    n = len(c)
    cand = np.zeros(n, bool)
    for di in range(119, n):
        jn = J[di]
        if np.isnan(jn) or jn >= 20: continue
        z = zd[di]
        if np.isnan(z) or not (l[di] <= z*1.001 and c[di] >= z*0.99): continue
        sc = 0
        jt = di >= 1 and J[di] > J[di-1]
        if jn < 12:
            sc += 5
            if jt: sc += 1
        else:
            sc += 3
            if jt: sc += 1
        trend = False
        if trend_system_A(l, h, it, ip, di): trend = True
        if ma5[di] > ma13[di] > ma40[di] > ma60[di]: trend = True
        if c[di] > ma20[di] > ma60[di]: trend = True
        if not np.isnan(zt[di]) and not np.isnan(zd[di]) and c[di] > zt[di] > zd[di]: trend = True
        if dif[di] > dea[di] and dif[di] > 0: trend = True
        if trend: sc += 2
        if c[di] > ma60[di]: sc += 1
        if chg[di] > 0 and vr5[di] > P["vol_confirm"]: sc += 1
        elif chg[di] < 0 and vr5[di] < 0.8: sc += 1
        if bull[di]: sc += 1
        if chg[di] < 0 and vr5[di] < 0.8: sc += 1
        if sc >= P["b1_th"]: cand[di] = True
    sig = np.zeros(n, bool); strong = np.zeros(n, bool)
    for di in np.where(cand)[0]:
        dn = di+1
        if dn >= n: continue
        if P["bull_gate"] and c[dn] <= o[dn]: continue
        sc = 2
        sc += 1
        sc += 2 if vr5[dn] > P["vr5"] else 1
        if J[dn] < P["j_max"]: sc += 1
        if chg[dn] > P["chg_hi"]: sc += 2
        elif chg[dn] > P["chg_lo"]: sc += 1
        if us[dn] < P["us"]: sc += 1
        if sc >= P["b2_th"]:
            sig[dn] = True
            if sc >= P["strong_th"]: strong[dn] = True
    return sig, strong

BASE_P = {"b1_th": 5, "b2_th": 5, "strong_th": 7, "vr5": 1.2, "chg_lo": 1.5, "chg_hi": 3.9,
          "j_max": 55, "us": 1.0, "bull_gate": True, "vol_confirm": 1.2}

# 分期基准(全股票)
base_by_period = {}
for pname in ["训练期", "验证期", "全部"]:
    vals = []
    for code, ft in feats.items():
        t1 = ft["t1_20"]; dates = ft["date"]
        for i in range(119, len(t1)):
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            if pname != "全部" and (("20250908" > ds) != (pname == "训练期")): continue
            vals.append(float(t1[i])*PCT)
    base_by_period[pname] = float(np.mean(vals))
print("基准(全股票): 训练%+.2f%% 验证%+.2f%% 全部%+.2f%%" % (base_by_period["训练期"], base_by_period["验证期"], base_by_period["全部"]))

def eval_variant(P):
    out = {"全部": [], "训练期": [], "验证期": []}; nsig = 0; nstrong = 0
    for code in codes:
        ft = feats[code]
        sig, strong = recompute(ft, P)
        t1 = ft["t1_20"]; dates = ft["date"]
        for i in np.where(sig)[0]:
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            per = "训练期" if ds < TRAIN_END else "验证期"
            out["全部"].append(float(t1[i])*PCT); out[per].append(float(t1[i])*PCT); nsig += 1
        nstrong += int(strong.sum())
    res = {"n_total": nsig, "n_strong": nstrong}
    for k, v in out.items():
        if v:
            v = np.array(v)
            res[k] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                      "win": float(100*(v > 0).mean()), "rel": float(v.mean())-base_by_period[k]}
    return res

variants = [("基准", BASE_P)]
for th in [4, 6, 7]:
    P = dict(BASE_P); P["b2_th"] = th
    variants.append(("b2_score>=%d" % th, P))
for th in [4, 6]:
    P = dict(BASE_P); P["b1_th"] = th
    variants.append(("b1_score>=%d" % th, P))
for vr in [1.0, 1.5]:
    P = dict(BASE_P); P["vr5"] = vr
    variants.append(("量比>%.1f" % vr, P))
P = dict(BASE_P); P["chg_lo"], P["chg_hi"] = 0.5, 2.0
variants.append(("涨幅门槛(0.5/2.0)", P))
P = dict(BASE_P); P["chg_lo"], P["chg_hi"] = 2.5, 5.0
variants.append(("涨幅门槛(2.5/5.0)", P))
for jm in [45, 65]:
    P = dict(BASE_P); P["j_max"] = jm
    variants.append(("确认日J<%d" % jm, P))
for us in [0.5, 2.0]:
    P = dict(BASE_P); P["us"] = us
    variants.append(("上影线<%.1f%%" % us, P))
P = dict(BASE_P); P["bull_gate"] = False
variants.append(("去掉阳线硬门禁", P))

print("\nOFAT(信号层, t1_20):")
print("%-18s %7s %7s %8s %8s %7s %8s %8s" % ("变体", "n", "强n", "mean%", "中位%", "win%", "训练rel", "验证rel"))
ofat = {}
for name, P in variants:
    r = eval_variant(P)
    ofat[name] = r
    print("%-18s %7d %7d %+8.2f %+8.2f %7.1f %+8.2f %+8.2f" % (
        name, r["n_total"], r["n_strong"], r["全部"]["mean"], r["全部"]["median"], r["全部"]["win"],
        r["训练期"]["rel"], r["验证期"]["rel"]))

# ---------- regime ----------
# pass1: 每日全市场等权收益
mkt_sum, mkt_cnt = {}, {}
for code, ft in feats.items():
    c = ft["c"].astype(np.float64); dates = ft["date"]
    for t in range(1, len(c)):
        d = dates[t].replace("-", "")
        if not (D0 <= d <= D1): continue
        if c[t-1] > 0:
            r = c[t]/c[t-1]-1
            mkt_sum[d] = mkt_sum.get(d, 0.0)+r; mkt_cnt[d] = mkt_cnt.get(d, 0)+1
mkt_ret = {d: mkt_sum[d]/mkt_cnt[d] for d in mkt_sum}
ds_list = sorted(mkt_ret)
mkt20, mktvol = {}, {}
for k, ds in enumerate(ds_list):
    if k >= 20:
        seg = [mkt_ret[x] for x in ds_list[k-20:k]]
        mkt20[ds] = sum(seg); mktvol[ds] = float(np.std(seg)*math.sqrt(244))
vs = sorted(mktvol.values()); t1, t2 = vs[len(vs)//3], vs[2*len(vs)//3]
# pass2: 按regime汇总信号与基准
regime = {"上涨": [], "下跌": [], "低波": [], "高波": []}
base_reg = {"上涨": [], "下跌": [], "低波": [], "高波": []}
for code, ft in feats.items():
    sb = ft["b2_sig"]; t1_ = ft["t1_20"]; dates = ft["date"]
    for i in range(119, len(t1_)):
        if np.isnan(t1_[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        m = mkt20.get(ds); v = mktvol.get(ds)
        if m is None: continue
        key = "上涨" if m > 0 else "下跌"
        regime[key].append(float(t1_[i])*PCT if sb[i] else np.nan); base_reg[key].append(float(t1_[i])*PCT)
        if v is not None:
            vk = "低波" if v < t1 else ("高波" if v > t2 else None)
            if vk:
                regime[vk].append(float(t1_[i])*PCT if sb[i] else np.nan); base_reg[vk].append(float(t1_[i])*PCT)
regime_out = {}
print("\nRegime(B2信号 vs 同regime基准):")
for k in ["上涨", "下跌", "低波", "高波"]:
    sv = np.array([x for x in regime[k] if not np.isnan(x)]); bv = np.array(base_reg[k])
    if len(sv) and len(bv):
        regime_out[k] = {"sig_n": len(sv), "sig_mean": float(sv.mean()), "sig_win": float(100*(sv > 0).mean()),
                         "base_n": len(bv), "base_mean": float(bv.mean()), "rel": float(sv.mean()-bv.mean())}
        print("  %s: 信号n=%6d mean=%+6.2f%% win%5.1f%% | 基准mean=%+6.2f%% | rel=%+6.2f" % (
            k, len(sv), sv.mean(), 100*(sv > 0).mean(), bv.mean(), sv.mean()-bv.mean()))

# 右截断
trunc = total = 0
for code, ft in feats.items():
    for i in np.where(ft["b2_sig"])[0]:
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        total += 1
        if np.isnan(ft["t1_20"][i]): trunc += 1
print("\n右截断: %d/%d" % (trunc, total))

json.dump({"ofat": ofat, "regime": regime_out, "base_by_period": base_by_period,
           "truncation": {"n_total": total, "n_truncated": trunc}},
          open(os.path.join(HERE, "b2_robust_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> b2_robust_results.json")
