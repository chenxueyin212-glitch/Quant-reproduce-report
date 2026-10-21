# -*- coding: utf-8 -*-
"""
B2因子重构评估 — Step1 特征预处理(基于 b1_features.pkl 增量计算)
B2 = 逆势确认: T日 B1候选(J<20 且 触知行多空线回收 且 b1_score>=5),
              T+1日 放量阳线确认(b2_score>=5; >=7为强信号)
口径与B1完全一致: 信号日收盘后可知 → 次日开盘入场 → 20个交易日后收盘出场
"""
import pickle, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_KLINES = 120
PCT = 100.0

t0 = time.time()
d = pickle.load(open(os.path.join(HERE, "b1_features.pkl"), "rb"))
feats1 = d["feats"]

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

def calc_b2_series(c, o, h, l, v):
    """B2依赖指标(除J/zx_t/zx_dk已在b1管道)"""
    n = len(c)
    ma5 = ma_cumsum(c, 5); ma13 = ma_cumsum(c, 13)
    ma20 = ma_cumsum(c, 20); ma40 = ma_cumsum(c, 40); ma60 = ma_cumsum(c, 60)
    ema12 = ema_series(c, 12); ema26 = ema_series(c, 26)
    dif = ema12 - ema26
    dea = np.full(n, np.nan)
    dif_start = 25
    if n > dif_start:
        dea[dif_start:] = ema_series(dif[dif_start:], 9)
    hist = 2*(dif - dea)
    vol_ma5 = ma_cumsum(v, 5)
    vr5 = np.where(vol_ma5 > 0, v/vol_ma5, 1.0)
    chg = np.zeros(n); chg[1:] = (c[1:]/c[:-1]-1)*PCT
    bullish = (c > o).astype(np.int8)
    upper_shadow = np.where(c > 0, (h-np.maximum(o, c))/c*PCT, 0.0)
    is_trough = np.zeros(n, bool); is_peak = np.zeros(n, bool)
    for i in range(3, n-3):
        if l[i] <= l[i-3:i+4].min(): is_trough[i] = True
        if h[i] >= h[i-3:i+4].max(): is_peak[i] = True
    return dict(ma5=ma5, ma13=ma13, ma20=ma20, ma40=ma40, ma60=ma60,
                dif=dif, dea=dea, hist=hist, vr5=vr5, chg=chg,
                bullish=bullish, upper_shadow=upper_shadow, is_trough=is_trough, is_peak=is_peak)

def trend_system_A(lows, highs, is_trough, is_peak, di):
    lo = max(0, di-56); hi = di-2
    if hi - lo < 2: return False
    troughs = lows[lo:hi][is_trough[lo:hi]]
    peaks = highs[lo:hi][is_peak[lo:hi]]
    if len(troughs) >= 2 and len(peaks) >= 2:
        return troughs[-1] > troughs[-2] and peaks[-1] > peaks[-2]
    return False

def b1_score(S, c, h, l, di):
    """check_b1 score(精确复现)"""
    J = S["J"]; j_now = J[di]
    score = 0
    j_turning = (di >= 1) and (J[di] > J[di-1])
    if j_now < 12:
        score += 5
        if j_turning: score += 1
    else:
        score += 3
        if j_turning: score += 1
    ma5, ma13, ma20 = S["ma5"][di], S["ma13"][di], S["ma20"][di]
    ma40, ma60 = S["ma40"][di], S["ma60"][di]
    trend = False
    if trend_system_A(l, h, S["is_trough"], S["is_peak"], di): trend = True
    if ma5 > ma13 > ma40 > ma60: trend = True
    if c[di] > ma20 > ma60: trend = True
    if not np.isnan(S["zx_t"][di]) and not np.isnan(S["zx_dk"][di]) and c[di] > S["zx_t"][di] > S["zx_dk"][di]: trend = True
    if S["dif"][di] > S["dea"][di] and S["dif"][di] > 0: trend = True
    if trend: score += 2
    if c[di] > ma60: score += 1
    chg = S["chg"][di]; vr5 = S["vr5"][di]
    if chg > 0 and vr5 > 1.2: score += 1
    elif chg < 0 and vr5 < 0.8: score += 1
    if S["bullish"][di]: score += 1
    if chg < 0 and vr5 < 0.8: score += 1
    return score

def b2_score(S, c, o, di):
    """check_b2 score(精确复现)"""
    score = 2
    if c[di] <= o[di]: return 0
    score += 1
    if S["vr5"][di] > 1.2: score += 2
    else: score += 1
    if S["J"][di] < 55: score += 1
    chg = S["chg"][di]
    if chg > 3.9: score += 2
    elif chg > 1.5: score += 1
    if S["upper_shadow"][di] < 1.0: score += 1
    return score

feats2 = {}
for idx, (code, ft) in enumerate(feats1.items()):
    c = ft["c"].astype(np.float64); o = ft["o"].astype(np.float64)
    h = ft["h"].astype(np.float64); l = ft["l"].astype(np.float64)
    v = ft["v"].astype(np.float64); n = len(c)
    if n < MIN_KLINES: continue
    S = calc_b2_series(c, o, h, l, v)
    S["J"] = ft["j"].astype(np.float64); S["zx_t"] = ft["zt"].astype(np.float64); S["zx_dk"] = ft["zd"].astype(np.float64)
    J = S["J"]; zd = S["zx_dk"]
    # B1候选
    cand = np.zeros(n, bool); cand_score = np.full(n, np.nan)
    for di in range(MIN_KLINES-1, n):
        jn = J[di]
        if np.isnan(jn) or jn >= 20: continue
        z = zd[di]
        if np.isnan(z) or not (l[di] <= z*1.001 and c[di] >= z*0.99): continue
        sc = b1_score(S, c, h, l, di)
        if sc >= 5:
            cand[di] = True; cand_score[di] = sc
    # B2确认(T+1)
    sig = np.zeros(n, bool); strong = np.zeros(n, bool); b2sc = np.full(n, np.nan)
    for di in np.where(cand)[0]:
        dn = di+1
        if dn >= n: continue
        sc = b2_score(S, c, o, dn)
        if sc >= 5:
            sig[dn] = True; b2sc[dn] = sc
            if sc >= 7: strong[dn] = True
    f2 = dict(ft)  # 复制B1全部字段(日期/K线/前瞻收益/后视特征)
    f2.update({k: (v.astype(np.float32) if v.dtype != bool else v) for k, v in S.items()})
    f2["b1_cand"] = cand; f2["b1_cand_score"] = cand_score.astype(np.float32)
    f2["b2_sig"] = sig; f2["b2_strong"] = strong; f2["b2_score"] = b2sc.astype(np.float32)
    feats2[code] = f2
    if (idx+1) % 500 == 0:
        print("  %d/%d  %.0fs" % (idx+1, len(feats1), time.time()-t0))

with open(os.path.join(HERE, "b2_features.pkl"), "wb") as f:
    pickle.dump({"feats": feats2}, f, protocol=4)
print("done: %d stocks, %.1f MB, %.0fs" % (len(feats2), os.path.getsize(os.path.join(HERE, "b2_features.pkl"))/1e6, time.time()-t0))
