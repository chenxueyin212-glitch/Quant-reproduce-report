# -*- coding: utf-8 -*-
"""
搬砖（砖型图VAR6A·多空临界）特征预处理 — 复刻 zge_compute.calc_zge_indicators + check_banzhuan
定义（check_banzhuan，2026-07-17 规则变更后）:
  VAR6A = SMA(SMA(X,6,1),6,1) - SMA(Y,4,1)
    X = (C-LLV(L,4))/(HHV(H,4)-LLV(L,4))*100
    Y = (HHV(H,4)-C)/(HHV(H,4)-LLV(L,4))*100 - 90
    注: 源码注释称 SMA(X,N,1)≈EMA α=1/N，实际实现 ema() 用 α=2/(N+1)——按代码复刻（偏离登记）
  红砖: VAR6A_t > VAR6A_{t-4}; 绿砖: <
  买入: 第1红砖 +5分 / 第2红砖 +3分（第3砖起弃用）
  趋势: C>MA60 +1
  支撑过滤(仅第1/2砖): 知行多空线 zd=(MA14+MA28+MA57+MA114)/4
    touch = low <= zd*1.001 且 recovery = C >= zd*0.99 → support_type="搬砖-多空临界", +2
    阳线(C>O)再 +1; 支撑未确认 → score 清零过滤
  生产信号(回测口径): score>=5 且 support_type=="搬砖-多空临界"
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20) — 与B1/B2/bp1/补票一致
复用 bp1_features.pkl 的 o/h/l/c/v 与路径指标(t1_20/mae20/...)。
末尾附 600 例校准（截断模式 vs 原引擎）。
"""
import pickle, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
ENGINE_DIR = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener"
D0, D1 = "20231101", "20260821"

src = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))
feats_in = src["feats"]
pool_codes = set(s["code"] for s in json.load(open(POOL, encoding="utf-8"))["stocks"])
print("宇宙 %d 只, 池 %d 只" % (len(feats_in), len(pool_codes)))


def ema(arr, period):
    """zge_compute.ema 逐位复刻: alpha=2/(period+1), 找首个连续 period 非NaN窗口初始化"""
    n_arr = len(arr)
    result = np.full(n_arr, np.nan)
    alpha = 2.0 / (period + 1)
    start_idx = -1
    for i in range(n_arr - period + 1):
        if not any(np.isnan(arr[i:i+period])):
            start_idx = i
            break
    if start_idx < 0:
        return result
    init_idx = start_idx + period - 1
    result[init_idx] = np.mean(arr[start_idx:start_idx+period])
    for i in range(init_idx + 1, n_arr):
        if np.isnan(arr[i]):
            result[i] = np.nan
            continue
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


def ma(arr, period):
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.mean(arr[i-period+1:i+1])
    return result


def indicators(o, h, l, c):
    """calc_zge_indicators 搬砖相关指标（MA14/28/57/114/60、zx趋势、VAR6A、红砖）"""
    n = len(c)
    ma14 = ma(c, 14); ma28 = ma(c, 28); ma57 = ma(c, 57); ma114 = ma(c, 114)
    ma60 = ma(c, 60)
    zx_dk = (ma14 + ma28 + ma57 + ma114) / 4.0

    # 知行趋势线 = EMA(EMA(C,10),10) — 复刻源码的 valid 剥离再 pad 写法
    ema10 = ema(c, 10)
    ema10_valid = ema10[~np.isnan(ema10)]
    zx_trend = ema(ema10_valid, 10)
    zx_trend_full = np.full(n, np.nan)
    valid_start = 18
    if len(zx_trend) > 0:
        trend_values = zx_trend[~np.isnan(zx_trend)]
        if len(trend_values) and valid_start + len(trend_values) <= n:
            zx_trend_full[valid_start:valid_start+len(trend_values)] = trend_values

    # VAR6A 砖型
    llv4 = np.full(n, np.nan); hhv4 = np.full(n, np.nan)
    for i in range(3, n):
        llv4[i] = np.min(l[i-3:i+1])
        hhv4[i] = np.max(h[i-3:i+1])
    rng4 = hhv4 - llv4
    x_raw = np.where(rng4 > 0, (c - llv4) / rng4 * 100, 50.0)
    y_raw = np.where(rng4 > 0, (hhv4 - c) / rng4 * 100 - 90, 10.0)
    x_valid = x_raw[3:]; y_valid = y_raw[3:]
    sma_x_6_valid = ema(x_valid, 6)
    sma_x_6_6_valid = ema(sma_x_6_valid, 6)
    sma_y_4_valid = ema(y_valid, 4)
    sma_x_6_6 = np.full(n, np.nan); sma_y_4 = np.full(n, np.nan)
    if len(sma_x_6_6_valid) > 0:
        sma_x_6_6[3:3+len(sma_x_6_6_valid)] = sma_x_6_6_valid
    if len(sma_y_4_valid) > 0:
        sma_y_4[3:3+len(sma_y_4_valid)] = sma_y_4_valid
    var6a = sma_x_6_6 - sma_y_4

    red_brick = np.zeros(n, dtype=bool); green_brick = np.zeros(n, dtype=bool)
    for i in range(4, n):
        if np.isnan(var6a[i]) or np.isnan(var6a[i-4]):
            continue
        if var6a[i] > var6a[i-4]:
            red_brick[i] = True
        elif var6a[i] < var6a[i-4]:
            green_brick[i] = True

    # 连续红砖数（从 i 往回数）
    consec = np.zeros(n, dtype=np.int32)
    run = 0
    for i in range(n):
        if red_brick[i]:
            run += 1
        else:
            run = 0
        consec[i] = run

    # chg_20d = closes[-1]/closes[-20] → 19 期（复刻源码 off-by-one，i>=20 才有值）
    chg20d = np.zeros(n)
    if n > 20:
        chg20d[20:] = (c[20:] / c[1:n-19] - 1.0) * 100.0

    return dict(zx_dk=zx_dk, zx_trend=zx_trend_full, var6a=var6a,
                red=red_brick, green=green_brick, consec=consec,
                ma60=ma60, chg20d=chg20d)


def score_series(ind, o, h, l, c):
    """check_banzhuan 逐条对应 → (score, sig, support_ok)"""
    n = len(c)
    score = np.zeros(n, dtype=np.int32)
    support_ok = np.zeros(n, dtype=bool)
    red = ind["red"]; consec = ind["consec"]
    zx_dk = ind["zx_dk"]; zx_trend = ind["zx_trend"]; ma60 = ind["ma60"]

    # ① 砖分
    score += np.where(red & (consec == 1), 5, 0)
    score += np.where(red & (consec == 2), 3, 0)
    # ② MA60
    score += np.where((~np.isnan(ma60)) & (c > ma60), 1, 0)

    # ③ 支撑过滤（仅第1/2红砖且双线有效）
    applicable = (~np.isnan(zx_trend)) & (~np.isnan(zx_dk)) & red & (consec <= 2)
    with np.errstate(invalid="ignore"):
        touch = l <= zx_dk * 1.001
        recovery = c >= zx_dk * 0.99
    ok = applicable & touch & recovery
    bullish = c > o
    score = np.where(applicable & ok, score + 2 + np.where(bullish, 1, 0), score)
    score = np.where(applicable & ~ok, 0, score)
    support_ok = ok

    sig = (score >= 5) & support_ok
    return score, sig, support_ok


feats_out = {}
n_sig = 0
for code, ft in feats_in.items():
    o = ft["o"].astype(np.float64); h = ft["h"].astype(np.float64)
    l = ft["l"].astype(np.float64); c = ft["c"].astype(np.float64)
    n = len(c)
    if n < 60:
        continue
    ind = indicators(o, h, l, c)
    score, sig, support_ok = score_series(ind, o, h, l, c)

    zd = ind["zx_dk"]
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_low = np.where(~np.isnan(zd) & (zd != 0), (l - zd) / np.abs(zd) * 100, np.nan)
        dist_close = np.where(~np.isnan(zd) & (zd != 0), (c - zd) / np.abs(zd) * 100, np.nan)

    feats_out[code] = {
        "date": np.array([str(x).replace("-", "") for x in ft["date"]]),  # D29 归一
        "sig": sig, "score": score, "consec": ind["consec"].astype(np.int8),
        "var6a": ind["var6a"].astype(np.float32),
        "zd": zd.astype(np.float32), "zt": ind["zx_trend"].astype(np.float32),
        "dist_low": dist_low.astype(np.float32), "dist_close": dist_close.astype(np.float32),
        "bullish": (c > o), "above_ma60": (~np.isnan(ind["ma60"])) & (c > ind["ma60"]),
        "chg20d": ind["chg20d"].astype(np.float32),
        "green": ind["green"],
        "t1_20": ft["t1_20"], "fwd20": ft["fwd20"],
        "mae20": ft["mae20"], "mfe20": ft["mfe20"],
        "pre20": ft["pre20"], "pre60": ft["pre60"],
        "dd20": ft["dd20"], "vol20": ft["vol20"],
        "below_days": ft["below_days"],
        "first5_min": ft["first5_min"], "last15_max": ft["last15_max"],
    }
    n_sig += int(sig.sum())

print("搬砖信号(score>=5+多空临界) %d 条" % n_sig)

# ================= 校准验证（截断模式 vs 原引擎） =================
sys.path.insert(0, ENGINE_DIR)
import zge_compute  # noqa

rng = np.random.default_rng(20261020)
codes = sorted(feats_in.keys())
sample_codes = [codes[i] for i in rng.choice(len(codes), 30, replace=False)]

# 20 个月末日（全样本日期轴）
alld = sorted({str(x) for ft in feats_in.values() for x in ft["date"]})
months = {}
for ds in alld:
    months[ds[:6]] = ds  # 每月最后一天
sample_dates = sorted(months.values())[-20:]

def to_klines(ft, upto):
    ds = ft["date"]; o = ft["o"]; h = ft["h"]; l = ft["l"]; c = ft["c"]; v = ft["v"]
    ks = []
    for i in range(len(ds)):
        if str(ds[i]) > upto:
            break
        ks.append({"open": float(o[i]), "high": float(h[i]), "low": float(l[i]),
                   "close": float(c[i]), "volume": float(v[i]), "date": str(ds[i])})
    return ks

n_cmp = 0; n_ok = 0; mismatch = []
for code in sample_codes:
    ft = feats_in[code]
    dates_norm = np.array([str(x).replace("-", "") for x in ft["date"]])
    ind_mine = indicators(ft["o"].astype(np.float64), ft["h"].astype(np.float64),
                          ft["l"].astype(np.float64), ft["c"].astype(np.float64))
    sc_mine, sig_mine, _ = score_series(ind_mine, ft["o"].astype(np.float64),
                                        ft["h"].astype(np.float64), ft["l"].astype(np.float64),
                                        ft["c"].astype(np.float64))
    for ds in sample_dates:
        idxs = np.where(dates_norm == ds)[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[-1])
        ks = to_klines(ft, ds)
        if len(ks) < 60:
            continue
        ind = zge_compute.calc_zge_indicators(ks)
        if ind is None:
            continue
        bz = zge_compute.check_banzhuan(ind)
        eng_score = int(bz.get("score", 0))
        eng_sig = bool(eng_score >= 5 and bz.get("details", {}).get("support_type") == "搬砖-多空临界")
        my_score = int(sc_mine[i]); my_sig = bool(sig_mine[i])
        n_cmp += 1
        if eng_score == my_score and eng_sig == my_sig:
            n_ok += 1
        else:
            mismatch.append({"code": code, "date": ds, "engine": (eng_score, eng_sig),
                             "mine": (my_score, my_sig)})

print("校准: %d/%d = %.2f%%" % (n_ok, n_cmp, 100.0*n_ok/max(n_cmp, 1)))
if mismatch[:5]:
    print("前5条不一致:", mismatch[:5])

pickle.dump({"feats": feats_out, "pool_codes": pool_codes},
            open(os.path.join(HERE, "banzhuan_features.pkl"), "wb"))
json.dump({"n_cmp": n_cmp, "n_ok": n_ok, "mismatch": mismatch[:20]},
          open(os.path.join(HERE, "banzhuan_verify.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved -> banzhuan_features.pkl / banzhuan_verify.json")
