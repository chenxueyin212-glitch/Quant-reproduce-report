# -*- coding: utf-8 -*-
"""
B1因子重构评估 — Step4 稳健性检验
1. 参数敏感性(OFAT): J阈值/ATR深度/收盘容差/触线容差/趋势线周期/多空线窗口
2. 2x2 条件交互: J<20 × 触线收稳
3. "不破"过滤器价值: 触线后收稳 vs 破位
4. Regime分析: 市场涨跌/波动率分位
5. 右截断: 末尾20根内信号计数
输出: b1_robust_results.json
"""
import pickle, os, json, math, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
BASE = {"j_th": 20, "atr_mult": 0.5, "close_tol": 0.99, "touch_tol": 1.001, "ema_p": 10, "dk_w": (14, 28, 57, 114)}

d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats, bench = d["feats"], d["bench"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
codes = list(feats.keys())

# 基准(同窗口同口径, 来自信号层)
sig_res = json.load(open(os.path.join(HERE, "b1_signal_results.json"), encoding="utf-8"))
BASE_MEAN = {"all": sig_res["naive"]["全股票等权基准"]["mean"],
             "训练期": sig_res["legacy_period"]["训练期"]["基准_全体"]["mean"],
             "验证期": sig_res["legacy_period"]["验证期"]["基准_全体"]["mean"]}

def ma_cumsum(arr, period):
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    r[period-1:] = (cs[period:] - cs[:-period]) / period
    return r

def ema_series(arr, period):
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    a = 2.0/(period+1); r[period-1] = np.mean(arr[:period])
    for i in range(period, n): r[i] = a*arr[i] + (1-a)*r[i-1]
    return r

def calc_j(closes, highs, lows):
    n = len(closes)
    low9 = np.full(n, np.nan); high9 = np.full(n, np.nan)
    for i in range(8, n):
        low9[i] = np.min(lows[i-8:i+1]); high9[i] = np.max(highs[i-8:i+1])
    denom = high9-low9
    rsv = np.where(denom > 0, (closes-low9)/denom*100, 50.0)
    k = np.full(n, 50.0); dd = np.full(n, 50.0)
    for i in range(9, n):
        k[i] = 2/3*k[i-1] + 1/3*rsv[i]; dd[i] = 2/3*dd[i-1] + 1/3*k[i]
    return 3*k-2*dd

def zx_trend_of(closes, p):
    e = ema_series(closes, p); ev = e[~np.isnan(e)]
    zt = ema_series(ev, p); out = np.full(len(closes), np.nan)
    ztv = zt[~np.isnan(zt)]
    if len(ztv): out[2*p-2:2*p-2+len(ztv)] = ztv
    return out

def zx_dk_of(closes, w):
    return sum(ma_cumsum(closes, x) for x in w)/len(w)

def detect(j, zt, zd, c, l, p):
    """返回 (sig_dc, sig_dk) bool 数组"""
    atr = 0.0214*c
    n = len(c); sdc = np.zeros(n, bool); sdk = np.zeros(n, bool)
    cond_j = (~np.isnan(j)) & (j < p["j_th"])
    tz = (~np.isnan(zt)) & (l <= zt*p["touch_tol"]) & (l >= (zt-p["atr_mult"]*atr)*0.995) & (c >= zt*p["close_tol"])
    td = (~np.isnan(zd)) & (l <= zd*p["touch_tol"]) & (l >= (zd-p["atr_mult"]*atr)*0.995) & (c >= zd*p["close_tol"])
    for i in range(59, n):
        if not cond_j[i]: continue
        if tz[i]: sdc[i] = True
        elif td[i]: sdk[i] = True
    return sdc | sdk, sdc, sdk

def eval_variant(p, recompute_lines=False):
    """重算信号并统计; recompute_lines=True 时重算趋势线/多空线"""
    out = {"all": [], "训练期": [], "验证期": []}
    n_sig = 0
    for code in codes:
        ft = feats[code]
        c = ft["c"].astype(np.float64); l = ft["l"].astype(np.float64); h = ft["h"].astype(np.float64)
        j = ft["j"].astype(np.float64)
        if recompute_lines:
            zt = zx_trend_of(c, p["ema_p"]); zd = zx_dk_of(c, p["dk_w"])
        else:
            zt = ft["zt"].astype(np.float64); zd = ft["zd"].astype(np.float64)
        sig, _, _ = detect(j, zt, zd, c, l, p)
        t1 = ft["t1_20"].astype(np.float64)*PCT
        dates = ft["date"]
        for i in np.where(sig)[0]:
            if np.isnan(t1[i]): continue
            ds = dates[i].replace("-", "")
            if not (D0 <= ds <= D1): continue
            per = "训练期" if ds < TRAIN_END else "验证期"
            out["all"].append(t1[i]); out[per].append(t1[i]); n_sig += 1
    res = {}
    for k, v in out.items():
        if v:
            v = np.array(v)
            res[k] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                      "win": float(100*(v > 0).mean()),
                      "rel": float(v.mean()) - BASE_MEAN[k]}
    res["n_total"] = n_sig
    return res

# ---------- 1. OFAT ----------
variants = []
variants.append(("基准", dict(BASE), False))
for th in [5, 10, 15, 25, 30]:
    p = dict(BASE); p["j_th"] = th
    variants.append(("J<%d" % th, p, False))
for m in [0.25, 1.0, 2.0]:
    p = dict(BASE); p["atr_mult"] = m
    variants.append(("ATR深度x%.2f" % m, p, False))
for ct in [0.98, 0.995, 0.999]:
    p = dict(BASE); p["close_tol"] = ct
    variants.append(("收盘容差%.3f" % ct, p, False))
for tt in [1.002, 1.005]:
    p = dict(BASE); p["touch_tol"] = tt
    variants.append(("触线容差%.3f" % tt, p, False))
for ep in [5, 20]:
    p = dict(BASE); p["ema_p"] = ep
    variants.append(("趋势线EMA(%d,%d)" % (ep, ep), p, True))
for w in [(10, 20, 60, 120), (5, 10, 20, 60)]:
    p = dict(BASE); p["dk_w"] = w
    variants.append(("多空线%s" % (w,), p, True))

print("参数敏感性(OFAT, t1_20信号层):")
print("%-18s %7s %8s %8s %7s %8s %8s %7s" % ("变体", "n", "mean%", "中位%", "win%", "训练rel", "验证rel", "验证win%"))
ofat = {}
for name, p, rc in variants:
    r = eval_variant(p, recompute_lines=rc)
    ofat[name] = r
    print("%-18s %7d %+8.2f %+8.2f %7.1f %+8.2f %+8.2f %7.1f" % (
        name, r["n_total"], r["all"]["mean"], r["all"]["median"], r["all"]["win"],
        r["训练期"]["rel"], r["验证期"]["rel"], r["验证期"]["win"]))

# ---------- 2. 2x2 条件交互 ----------
cells = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
for code in codes:
    ft = feats[code]
    cj = ft["cond_j"]; hz = ft["hold_zt"]
    t1 = ft["t1_20"].astype(np.float64)*PCT; dates = ft["date"]
    for i in range(59, len(t1)):
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        cells[(int(cj[i]), int(hz[i]))].append(t1[i])
interact = {}
print("\n2x2条件交互(t1_20):")
for (a, b), v in cells.items():
    if v:
        v = np.array(v)
        interact["J<20=%d,触线收稳=%d" % (a, b)] = {"n": len(v), "mean": float(v.mean()),
                                                    "median": float(np.median(v)), "win": float(100*(v > 0).mean())}
        print("  J<20=%d 触线收稳=%d  n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (a, b, len(v), v.mean(), np.median(v), 100*(v > 0).mean()))

# ---------- 3. 不破过滤器价值 ----------
nb = {"收稳": [], "破位": []}
for code in codes:
    ft = feats[code]
    cj = ft["cond_j"]; tz = ft["touch_zt"]; hz = ft["hold_zt"]; bz = ft["break_zt"]
    t1 = ft["t1_20"].astype(np.float64)*PCT; dates = ft["date"]
    for i in range(59, len(t1)):
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        if cj[i] and hz[i]: nb["收稳"].append(t1[i])
        elif cj[i] and bz[i]: nb["破位"].append(t1[i])
nob = {}
print("\n'不破'过滤器(J<20且触趋势线):")
for k, v in nb.items():
    if v:
        v = np.array(v)
        nob[k] = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)), "win": float(100*(v > 0).mean())}
        print("  %s n=%7d mean=%+7.2f%% median=%+7.2f%% win=%5.1f%%" % (k, len(v), v.mean(), np.median(v), 100*(v > 0).mean()))

# ---------- 4. Regime ----------
# 等权全A日收益 -> 市场20日收益/波动率
all_dates = sorted({ds.replace("-", "") for ft in feats.values() for ds in ft["date"]})
mkt_ret = {}
for ds in all_dates:
    rs = []
    for code, ft in feats.items():
        dates = list(ft["date"])
        try: i = dates.index(ds.replace("-", ""))
        except ValueError: continue
        if i == 0: continue
        c0, c1 = float(ft["c"][i-1]), float(ft["c"][i])
        if c0 > 0: rs.append(c1/c0-1)
    if rs: mkt_ret[ds] = float(np.mean(rs))
ds_list = sorted(mkt_ret)
mkt20 = {}; mktvol = {}
for k, ds in enumerate(ds_list):
    if k >= 20:
        seg = [mkt_ret[x] for x in ds_list[k-20:k]]
        mkt20[ds] = sum(seg)
        mktvol[ds] = float(np.std(seg)*math.sqrt(244))
vol_sorted = sorted(mktvol.values())
vol_t1, vol_t2 = vol_sorted[len(vol_sorted)//3], vol_sorted[2*len(vol_sorted)//3]
regime = {"上涨": [], "下跌": [], "低波": [], "高波": []}
base_regime = {"上涨": [], "下跌": [], "低波": [], "高波": []}
for code in codes:
    ft = feats[code]
    sb = ft["sig_base"]; t1 = ft["t1_20"].astype(np.float64)*PCT; dates = ft["date"]
    for i in range(59, len(t1)):
        if np.isnan(t1[i]): continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        m = mkt20.get(ds); v = mktvol.get(ds)
        if m is None: continue
        key = "上涨" if m > 0 else "下跌"
        regime[key].append(t1[i] if sb[i] else np.nan)
        base_regime[key].append(t1[i])
        if v is not None:
            vk = "低波" if v < vol_t1 else ("高波" if v > vol_t2 else None)
            if vk:
                regime[vk].append(t1[i] if sb[i] else np.nan)
                base_regime[vk].append(t1[i])
regime_out = {}
print("\nRegime分析(B1信号 vs 同regime基准, t1_20):")
for k in ["上涨", "下跌", "低波", "高波"]:
    sv = np.array([x for x in regime[k] if not np.isnan(x)])
    bv = np.array(base_regime[k])
    if len(sv) and len(bv):
        regime_out[k] = {"sig_n": len(sv), "sig_mean": float(sv.mean()), "sig_win": float(100*(sv > 0).mean()),
                         "base_n": len(bv), "base_mean": float(bv.mean()),
                         "rel": float(sv.mean()-bv.mean())}
        print("  %s: 信号n=%6d mean=%+6.2f%% win%5.1f%% | 基准n=%7d mean=%+6.2f%% | 相对%+6.2f pct" % (
            k, len(sv), sv.mean(), 100*(sv > 0).mean(), len(bv), bv.mean(), sv.mean()-bv.mean()))

# ---------- 5. 右截断 ----------
trunc = 0; total = 0
for code in codes:
    ft = feats[code]; sb = ft["sig_base"]; t1 = ft["t1_20"]; dates = ft["date"]
    for i in np.where(sb > 0)[0]:
        if i < 59: continue
        ds = dates[i].replace("-", "")
        if not (D0 <= ds <= D1): continue
        total += 1
        if np.isnan(t1[i]): trunc += 1
print("\n右截断: 信号%d条中%d条无完整20日前瞻(已排除于上述统计)" % (total, trunc))

json.dump({"ofat": ofat, "interact": interact, "no_break": nob, "regime": regime_out,
           "truncation": {"n_total": total, "n_truncated": trunc}, "base_mean": BASE_MEAN},
          open(os.path.join(HERE, "b1_robust_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> b1_robust_results.json")
