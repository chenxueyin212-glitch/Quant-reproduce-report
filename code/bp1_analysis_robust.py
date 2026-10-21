# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step4 稳健性检验
OFAT(技术五条件 + L0门槛 + 持有期) / regime(趋势+波动) / 右截断
bp1特有两项: ①L0门槛敏感性(40->35/45/50) ②持有期(10/20/30日)
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
codes = list(feats.keys())

BASE_P = {"l0": 40, "range": 25, "atr": 0.06, "brk": 3, "vol5": 1.5, "vol60": 1.5, "hold": 20}

def variant_sig(ft, P):
    """按参数P重算bp1信号掩码(用pkl中预存的条件分解量)"""
    n = len(ft["date"])
    pr = ft["platform_range"]; va = ft["vol_atr_ratio"]; ba = ft["breakout_amp"]
    v5 = ft["vr5_1"]; v60 = ft["vr_long"]; nr = ft["n_raw"]
    with np.errstate(invalid="ignore"):
        m = (np.arange(n) >= 59) & (pr < P["range"]) & (va < P["atr"]) & (ba >= P["brk"]) \
            & (v5 > P["vol5"]) & (v60 > P["vol60"]) & (nr >= P["l0"])
    return m

def pnl_of_hold(ft, i, hold):
    """T+1开盘入场, 持有hold个交易日的收盘价收益"""
    o = ft["o"].astype(np.float64); c = ft["c"].astype(np.float64); n = len(c)
    e = i+1; x = i+hold
    if e >= n or x >= n or o[e] <= 0 or c[x] <= 0: return None
    return float(c[x]/o[e]-1)*PCT

# 分期基准(全股票, hold=20)
base_by_period = {}
for pname in ["训练期", "验证期", "全部"]:
    vals = []
    for code, ft in feats.items():
        t1 = ft["t1_20"]; dates = ft["date"]
        for i in range(59, len(t1)):
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            if pname != "全部" and (("20250908" > ds) != (pname == "训练期")): continue
            vals.append(float(t1[i])*PCT)
    base_by_period[pname] = float(np.mean(vals))
print("基准(全股票, hold=20): 训练%+.2f%% 验证%+.2f%% 全部%+.2f%%" % (
    base_by_period["训练期"], base_by_period["验证期"], base_by_period["全部"]))

def eval_variant(P):
    out = {"全部": [], "训练期": [], "验证期": []}; nsig = 0
    for code in codes:
        ft = feats[code]
        m = variant_sig(ft, P)
        t1 = ft["t1_20"]; dates = ft["date"]
        hold = P["hold"]
        for i in np.where(m)[0]:
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            pnl = pnl_of_hold(ft, i, hold) if hold != 20 else (None if np.isnan(t1[i]) else float(t1[i])*PCT)
            if pnl is None: continue
            per = "训练期" if ds < TRAIN_END else "验证期"
            out["全部"].append(pnl); out[per].append(pnl); nsig += 1
    res = {"n_total": nsig}
    for k, v in out.items():
        if v:
            v = np.array(v)
            res[k] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                      "win": float(100*(v > 0).mean()), "rel": float(v.mean())-base_by_period[k]}
    return res

variants = [("基准", dict(BASE_P))]
for l0 in [35, 45, 50]:
    P = dict(BASE_P); P["l0"] = l0
    variants.append(("L0门槛n_raw>=%d" % l0, P))
for rg in [20, 30]:
    P = dict(BASE_P); P["range"] = rg
    variants.append(("平台振幅<%d%%" % rg, P))
for at in [0.05, 0.07]:
    P = dict(BASE_P); P["atr"] = at
    variants.append(("ATR/MA<%.2f" % at, P))
for bk in [2, 4]:
    P = dict(BASE_P); P["brk"] = bk
    variants.append(("突破>=%d%%" % bk, P))
for vv in [1.3, 1.8]:
    P = dict(BASE_P); P["vol5"] = vv; P["vol60"] = vv
    variants.append(("量比>%.1f(双)" % vv, P))
P = dict(BASE_P); P["vol5"] = 1.3
variants.append(("量比>1.3(仅5日)", P))
P = dict(BASE_P); P["vol60"] = 1.3
variants.append(("量比>1.3(仅60日)", P))
for hd in [10, 30]:
    P = dict(BASE_P); P["hold"] = hd
    variants.append(("持有%d日" % hd, P))

print("\nOFAT(信号层, 对应持有期收益):")
print("%-18s %7s %8s %8s %7s %9s %9s" % ("变体", "n", "mean%", "中位%", "win%", "训练rel", "验证rel"))
ofat = {}
for name, P in variants:
    r = eval_variant(P)
    ofat[name] = r
    print("%-18s %7d %+8.2f %+8.2f %7.1f %+9.2f %+9.2f" % (
        name, r["n_total"], r["全部"]["mean"], r["全部"]["median"], r["全部"]["win"],
        r["训练期"]["rel"], r["验证期"]["rel"]))

# ---------- regime ----------
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
vs = sorted(mktvol.values()); t1v, t2v = vs[len(vs)//3], vs[2*len(vs)//3]
regime = {"上涨": [], "下跌": [], "低波": [], "高波": []}
base_reg = {"上涨": [], "下跌": [], "低波": [], "高波": []}
for code, ft in feats.items():
    sb = ft["bp1_sig"]; t1_ = ft["t1_20"]; dates = ft["date"]
    for i in range(59, len(t1_)):
        if np.isnan(t1_[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        m = mkt20.get(ds); v = mktvol.get(ds)
        if m is None: continue
        key = "上涨" if m > 0 else "下跌"
        regime[key].append(float(t1_[i])*PCT if sb[i] else np.nan); base_reg[key].append(float(t1_[i])*PCT)
        if v is not None:
            vk = "低波" if v < t1v else ("高波" if v > t2v else None)
            if vk:
                regime[vk].append(float(t1_[i])*PCT if sb[i] else np.nan); base_reg[vk].append(float(t1_[i])*PCT)
regime_out = {}
print("\nRegime(bp1信号 vs 同regime全股票基准):")
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
    for i in np.where(ft["bp1_sig"])[0]:
        ds = ft["date"][i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        total += 1
        if np.isnan(ft["t1_20"][i]): trunc += 1
print("\n右截断: %d/%d" % (trunc, total))

json.dump({"ofat": ofat, "regime": regime_out, "base_by_period": base_by_period,
           "truncation": {"n_total": total, "n_truncated": trunc}},
          open(os.path.join(HERE, "bp1_robust_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> bp1_robust_results.json")
