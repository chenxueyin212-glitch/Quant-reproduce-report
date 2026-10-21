# -*- coding: utf-8 -*-
"""
bp1 校准验证 v2 — 截断模式金标准比对(修复后)

与 bp1_crosscheck.py 的区别:
  crosscheck 的 "mine" 取自全序列 pkl(存在d2C部分日前视, 已修);
  verify2 的 "mine" 在**截断到采样日**的序列上用修复后代码现算, 与原引擎数据口径完全一致。
覆盖面: 40只池内票 x 25个月首日 = 1000 组比对。

my_per_stock 块逐行复制自 bp1_prep.py 的 compute_stock 逻辑(修复后版本):
  - d2 C部分: pos 上锚=窗口末日(消除日前视)
  - RPS: 截面=所有 len>=rps_period 的票
  - n_raw: float64
"""
import os, sys, json, sqlite3, random, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _niigu_judge_ref import compute_niugu_score
from _shunshi_buy_points_ref import _bp1_platform_breakout, _check_l2_daily_gate

SRC = r"C:/Users/Lenovo/WorkBuddy/2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
t0 = time.time()

MIN_KLINES = 60

# ================= 工具(复制自 bp1_prep.py, 含全部修复) =================
def ma_arr(arr, period):
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    r[period-1:] = np.lib.stride_tricks.sliding_window_view(arr, period).mean(axis=1)
    return r

def atr_wilder(highs, lows, closes, period=14):
    n = len(closes); tr = np.zeros(n)
    tr[0] = highs[0]-lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr

def rolling_any(a, w):
    n = len(a); r = np.zeros(n, bool)
    if n < w: return r
    cs = np.concatenate([[0], np.cumsum(a.astype(np.int32))])
    r[w-1:] = (cs[w:] - cs[:-w]) > 0
    return r

def win_corr_upto(x, y, w):
    n = len(x); r = np.full(n, np.nan)
    if n < w: return r
    svx = np.lib.stride_tricks.sliding_window_view(x, w)
    svy = np.lib.stride_tricks.sliding_window_view(y, w)
    mx = svx.mean(axis=1); my = svy.mean(axis=1)
    cx = svx-mx[:, None]; cy = svy-my[:, None]
    cov = (cx*cy).sum(axis=1)
    den = np.sqrt((cx**2).sum(axis=1)*(cy**2).sum(axis=1))
    ok = den > 0
    r[w-1:] = np.where(ok, cov/np.where(ok, den, 1), np.nan)
    return r

def rolling_pct_of_last(series, w, min_n=None):
    n = len(series); r = np.full(n, np.nan)
    if n < 10: return r
    pad = np.concatenate([np.full(w-1, np.nan), series])
    sv = np.lib.stride_tricks.sliding_window_view(pad, w)
    cur = series
    valid = ~np.isnan(sv)
    nvalid = valid.sum(axis=1)
    cnt = ((sv <= cur[:, None]) & valid).sum(axis=1)
    ok = (nvalid >= 10) & (~np.isnan(cur))
    with np.errstate(invalid="ignore"):
        r[:] = np.where(ok, cnt/np.where(nvalid > 0, nvalid, 1)*100, np.nan)
    if min_n is not None:
        r[:min_n-1] = np.nan
    return r

# ================= per-stock 特征(复制自 bp1_prep.py 修复版) =================
def compute_stock(rows):
    dates = np.array([r[0] for r in rows])
    o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
    l = np.array([float(r[3]) for r in rows]); c = np.array([float(r[4]) for r in rows])
    v = np.array([float(r[5]) for r in rows]); n = len(c)
    ma5 = ma_arr(c, 5); ma10 = ma_arr(c, 10); ma20 = ma_arr(c, 20); ma60 = ma_arr(c, 60)
    vma5_1 = np.concatenate([[np.nan], ma_arr(v, 5)[:-1]])
    vma60_1 = np.full(n, np.nan)
    for t in range(1, n):
        vma60_1[t] = (v[t-60:t].mean() if t >= 61 else v[:t].mean())
    vma20 = ma_arr(v, 20)
    vr5_today = v/np.where(ma_arr(v, 5) > 0, ma_arr(v, 5), np.nan)
    atr14 = atr_wilder(h, l, c, 14)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr_simple = ma_arr(tr, 14)
    ma_c14 = ma_arr(c, 14)
    std20 = np.full(n, np.nan)
    if n >= 20:
        std20[19:] = np.lib.stride_tricks.sliding_window_view(c, 20).std(axis=1)
    bbw = np.full(n, np.nan)
    okb = (~np.isnan(ma20)) & (~np.isnan(std20)) & (ma20 > 0)
    upper = ma20 + 2*std20; lower = ma20 - 2*std20
    bbw[okb] = (upper[okb]-lower[okb])/ma20[okb]
    atr_ratio = np.full(n, np.nan)
    ok2 = (~np.isnan(atr_simple)) & (c > 0)
    atr_ratio[ok2] = atr_simple[ok2]/c[ok2]
    bbw_pct = rolling_pct_of_last(bbw, 120, min_n=30)
    atr_pct = rolling_pct_of_last(atr_ratio, 120, min_n=24)
    # ---- dim1 ----
    d1 = np.zeros(n)
    perf = (~np.isnan(ma5)) & (~np.isnan(ma10)) & (~np.isnan(ma20)) & (~np.isnan(ma60)) & \
           (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
    tri = (~np.isnan(ma5)) & (~np.isnan(ma20)) & (~np.isnan(ma60)) & (ma5 > ma20) & (ma20 > ma60) & ~perf
    base = (~np.isnan(ma20)) & (~np.isnan(ma60)) & (c > ma20) & (ma20 > ma60) & ~tri & ~perf
    nb = (~np.isnan(ma60)) & (c > ma60) & ~tri & ~base & ~perf
    d1[perf] = 15; d1[tri] = 12; d1[base] = 8; d1[nb] = 4
    ma20_slope = np.full(n, np.nan)
    if n >= 25:
        ma20_5ago = np.concatenate([[np.nan]*5, ma20[:-5]])
        ma20_slope = np.where((ma20_5ago > 0) & (~np.isnan(ma20_5ago)), (ma20/ma20_5ago-1)/5*100, np.nan)
    d1 += np.where(ma20_slope > 0.3, 5, np.where(ma20_slope > 0, 3, 0))
    ma60_slope = np.full(n, np.nan)
    if n >= 65:
        ma60_5ago = np.concatenate([[np.nan]*5, ma60[:-5]])
        ma60_slope = np.where((ma60_5ago > 0) & (~np.isnan(ma60_5ago)), (ma60/ma60_5ago-1)/5*100, np.nan)
    d1 += np.where(ma60_slope > 0.1, 3, np.where(ma60_slope > 0, 2, 0))
    spread_now = (ma5-ma60)/np.where(ma60 > 0, ma60, np.nan)
    spread_5ago = np.full(n, np.nan)
    if n >= 25:
        ma5_5ago = np.concatenate([[np.nan]*5, ma5[:-5]])
        ma60_5ago2 = np.concatenate([[np.nan]*5, ma60[:-5]])
        spread_5ago = (ma5_5ago-ma60_5ago2)/np.where(ma60_5ago2 > 0, ma60_5ago2, np.nan)
    d1 += np.where(perf & (spread_now > spread_5ago+0.01), 2, 0)
    d1 = np.minimum(d1, 25)
    # ---- dim2 ----
    d2 = np.zeros(n)
    shrink = (v < vma20*0.6) & (~np.isnan(vma20))
    has_shrink = rolling_any(shrink, 20)
    d2 += np.where(has_shrink & (vr5_today > 1.5), 10, np.where(has_shrink & (vr5_today > 1.2), 7, np.where(has_shrink, 4, 0)))
    dv = np.diff(v, prepend=np.nan); dc = np.diff(c, prepend=np.nan)
    vp_corr = win_corr_upto(dv, dc, 5)
    d2 += np.where(vp_corr > 0.5, 5, np.where(vp_corr > 0, 3, 0))
    bullish = c > o
    lvr = v/np.where(vma20 > 0, vma20, np.nan)
    is_b5 = bullish & (lvr > 2.0) & (~np.isnan(lvr))
    is_b15 = bullish & (lvr > 1.5) & (~np.isnan(lvr))
    bottom_score = np.zeros(n)
    if n >= 60:
        am = np.lib.stride_tricks.sliding_window_view(l, 60).argmin(axis=1)
        gidx = np.arange(59, n)-59+am
        found = np.zeros(n-59, bool); score = np.zeros(n-59)
        jend = np.arange(59, n)
        for off in (-1, 0, 1, 2, 3):
            pos = np.clip(gidx+off, 0, jend)
            hit5 = is_b5[pos] & (~found)
            score[hit5] = 5; found |= hit5
            hit15 = is_b15[pos] & (~found)
            score[hit15] = 3; found |= hit15
        bottom_score[59:] = score
    d2 += bottom_score
    d2 = np.minimum(d2, 20)
    # ---- dim4 ----
    d4 = np.where(bbw_pct <= 10, 10, np.where(bbw_pct <= 20, 8, np.where(bbw_pct <= 30, 6,
        np.where(bbw_pct <= 50, 4, np.where(bbw_pct <= 80, 2, np.where(~np.isnan(bbw_pct), 0, 0))))))
    d4 += np.where(atr_pct <= 10, 5, np.where(atr_pct <= 20, 4, np.where(atr_pct <= 30, 3,
        np.where(atr_pct <= 50, 2, np.where(atr_pct <= 80, 1, np.where(~np.isnan(atr_pct), 0, 0))))))
    d4 = np.minimum(d4, 15)
    # ---- dim3 extra ----
    d3_extra = np.zeros(n)
    if n >= 60:
        high60 = np.full(n, np.nan)
        high60[59:] = np.lib.stride_tricks.sliding_window_view(c, 60).max(axis=1)
        d3_extra += np.where((~np.isnan(high60)) & (c >= high60*0.98), 2, 0)
    chg5 = np.full(n, np.nan)
    if n >= 6:
        chg5[5:] = (c[5:]/c[:-5]-1)*100
    d3_extra += np.where(chg5 > 3, 2, 0)
    d6 = np.full(n, 6.0)
    # ---- stage_mult ----
    ext = c/np.where(ma60 > 0, ma60, np.nan)-1.0
    ext_mult = np.where(ext <= 0.20, 1.0, np.where(ext >= 0.50, 0.82,
               np.where(~np.isnan(ext), 1.0-(ext-0.20)/0.30*0.18, 1.0)))
    up = (ma20 > ma60) & (~np.isnan(ma20)) & (~np.isnan(ma60))
    valid_prev = (~np.isnan(np.concatenate([[np.nan], ma20[:-1]]))) & (~np.isnan(np.concatenate([[np.nan], ma60[:-1]])))
    cross = np.zeros(n, bool)
    cross[1:] = up[1:] & (~up[:-1]) & valid_prev[1:]
    last_cross = np.maximum.accumulate(np.where(cross, np.arange(n), -1))
    trend_age = np.where(last_cross >= 0, np.arange(n)-last_cross, np.nan)
    age_mult = np.ones(n)
    age_mult[up & (~np.isnan(trend_age)) & (trend_age >= 120)] = 0.85
    age_mult[up & (~np.isnan(trend_age)) & (trend_age >= 60) & (trend_age < 120)] = 0.90
    stage_mult = np.minimum(ext_mult, age_mult)
    # ---- bp1 技术五条件 ----
    cond_len = np.zeros(n, bool); cond_len[MIN_KLINES-1:] = True
    ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    if n >= 21:
        ph[20:] = np.lib.stride_tricks.sliding_window_view(h, 20).max(axis=1)[:-1]
        pl[20:] = np.lib.stride_tricks.sliding_window_view(l, 20).min(axis=1)[:-1]
    range_pct = (ph-pl)/np.where(pl > 0, pl, np.nan)*100
    cond_platform = (~np.isnan(range_pct)) & (range_pct < 25)
    vr_atr = atr14/np.where(ma_c14 > 0, ma_c14, np.nan)
    cond_vol = (~np.isnan(vr_atr)) & (vr_atr < 0.06)
    brk = c/np.where(ph > 0, ph, np.nan)-1.0
    cond_break = (~np.isnan(brk)) & (c > ph) & (brk*100 >= 3)
    cond_vol5 = (~np.isnan(vma5_1)) & (v > vma5_1*1.5)
    cond_vol60 = (~np.isnan(vma60_1)) & (v > vma60_1*1.5)
    bp1_tech = cond_len & cond_platform & cond_vol & cond_break & cond_vol5 & cond_vol60
    # ---- L2 ----
    l2_ma20_up = (~np.isnan(ma20_slope)) & (ma20_slope >= -0.5)
    l2_above = c > ma20
    rsi = np.full(n, np.nan)
    if n > 14:
        deltas = np.diff(c); gains = np.where(deltas > 0, deltas, 0); losses = np.where(deltas < 0, -deltas, 0)
        ag = np.full(n, np.nan); al = np.full(n, np.nan)
        ag[14] = np.mean(gains[:14]); al[14] = np.mean(losses[:14])
        for i in range(15, n):
            ag[i] = (ag[i-1]*13+gains[i-1])/14; al[i] = (al[i-1]*13+losses[i-1])/14
        rs = np.where(al > 0, ag/al, 0)
        rsi = 100-100/(1+rs)
    l2_div = np.zeros(n, bool)
    if n >= 34:
        off = np.lib.stride_tricks.sliding_window_view(h, 20).argmax(axis=1)
        idx_hi = np.clip(np.arange(19, n)-19+off, 0, n-1)
        rhp = c[idx_hi]; rhr = rsi[idx_hi]
        l2_div[19:] = (c[19:] >= rhp*0.99) & (rsi[19:] < rhr-5)
    l2_pass = l2_ma20_up & l2_above & (~l2_div)
    return {"d1": d1, "d2": d2, "d4": d4, "d6": d6, "d3_extra": d3_extra,
            "stage_mult64": stage_mult, "bp1_tech": bp1_tech, "l2_pass": l2_pass}

def my_rps(closes_q, ref_map):
    """截断模式RPS(修复后口径): p=min(250,n); 截面=所有len>=p的票; 收益按p算; 百分位=mean(<=)"""
    n = len(closes_q)
    if len(ref_map) < 10:
        return None
    p = min(250, n)
    my = closes_q[-1]/closes_q[-p]-1.0
    rets = np.array([cc[-1]/cc[-p]-1.0 for cc in ref_map.values() if len(cc) >= p])
    if len(rets) == 0:
        return None
    return 100.0*float(np.mean(rets <= my))

def my_d3(rps, extra):
    if rps is None:
        base = 0
    elif rps >= 90: base = 20
    elif rps >= 80: base = 16
    elif rps >= 70: base = 12
    elif rps >= 50: base = 8
    elif rps >= 30: base = 4
    else: base = 0
    return min(base+extra, 20)

# ================= 采样与比对 =================
POOL_FILE = os.path.join(HERE, "_pool_20260804.json")
pool_codes = sorted(set(s["code"] for s in json.load(open(POOL_FILE, encoding="utf-8"))["stocks"]))
db_codes = set(r[0] for r in cur.execute("SELECT DISTINCT code FROM klines"))
ref_pool = [c for c in pool_codes if c in db_codes]
print("池内可用 %d 只" % len(ref_pool))

def db_klines(code, upto=None):
    q = "SELECT date,open,high,low,close,volume FROM klines WHERE code=?"
    args = [code]
    if upto: q += " AND date<=?"; args.append(upto)
    q += " ORDER BY date"
    rows = cur.execute(q, args).fetchall()
    return [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]

random.seed(11)
sample_codes = random.sample(ref_pool, 40)
# 年轻股(上市晚, DB行数<200): 测 d4 扩展窗口(n<120) + RPS 年轻票路径(p<250)
row_counts = {c: len(db_klines(c)) for c in ref_pool}
young_pool = [c for c in ref_pool if 60 <= row_counts[c] < 200]
random.seed(23)
sample_codes += random.sample(young_pool, min(10, len(young_pool)))
print("年轻股候选 %d 只, 采样 %d 只" % (len(young_pool), min(10, len(young_pool))))
# 预计算全部池内票的K线(内存中截断, 避免重复查DB)
klines_all = {c: db_klines(c) for c in ref_pool}
# 采样日: 2024-06 ~ 2026-06 每月首个交易日
all_dates = sorted({d for c in sample_codes for d, *_ in klines_all[c]})
month_first = {}
for dt in all_dates:
    ym = dt[:6]
    if ym not in month_first: month_first[ym] = dt
sample_dates = [v for ym, v in sorted(month_first.items()) if "202406" <= ym <= "202606"]
print("采样: %d 只 x %d 日" % (len(sample_codes), len(sample_dates)))

n_cmp = 0; n_ok = 0; mism = []
for code in sample_codes:
    kls = klines_all[code]
    for sd in sample_dates:
        sub = [k for k in kls if k[0] <= sd]
        if len(sub) < 60: continue
        # --- mine: 截断序列现算 ---
        ft = compute_stock(sub)
        i = len(sub)-1
        ref_map = {}
        for rc in ref_pool:
            rsub = [k for k in klines_all[rc] if k[0] <= sd]
            if len(rsub) >= 20:
                ref_map[rc] = np.array([k[4] for k in rsub])
        closes_q = np.array([k[4] for k in sub])
        rps = my_rps(closes_q, ref_map)
        d3 = my_d3(rps, float(ft["d3_extra"][i]))
        s = float(ft["d1"][i]+ft["d2"][i]+d3+ft["d4"][i]+ft["d6"][i])
        n_raw = int(np.floor(s*float(ft["stage_mult64"][i])))
        mine = {"d1": float(ft["d1"][i]), "d2": float(ft["d2"][i]), "d3": float(d3),
                "d4": float(ft["d4"][i]), "d6": float(ft["d6"][i]), "n_raw": n_raw,
                "tech": bool(ft["bp1_tech"][i]), "l2": bool(ft["l2_pass"][i])}
        # --- theirs: 原引擎 ---
        kdict = [{"date": k[0], "open": k[1], "high": k[2], "low": k[3], "close": k[4], "volume": k[5]} for k in sub]
        nr = compute_niugu_score(kdict, all_closes_map=ref_map)
        bp = _bp1_platform_breakout(kdict)
        l2 = _check_l2_daily_gate(kdict)
        theirs = {"d1": nr["dimensions"]["趋势结构"]["score"], "d2": nr["dimensions"]["量价关系"]["score"],
                  "d3": nr["dimensions"]["相对强弱"]["score"], "d4": nr["dimensions"]["波动率压缩"]["score"],
                  "d6": nr["dimensions"]["筹码结构"]["score"], "n_raw": nr["n_raw_score"],
                  "tech": bp is not None, "l2": l2[0]}
        n_cmp += 1
        bad = [k for k in mine if mine[k] != theirs[k]]
        if bad:
            mism.append((code, sd, bad, {k: (mine[k], theirs[k]) for k in bad}))
        else:
            n_ok += 1
    print("  %s done (累计 %d/%d, %.0fs)" % (code, n_ok, n_cmp, time.time()-t0))

print("\n===== 截断模式比对(修复后) =====")
print("比对样本: %d | 全维度一致: %d (%.2f%%)" % (n_cmp, n_ok, 100.0*n_ok/max(n_cmp, 1)))
for k in ["d1", "d2", "d3", "d4", "d6", "n_raw", "tech", "l2"]:
    nk = sum(1 for m in mism if k in m[2])
    print("  %-6s 不一致 %d" % (k, nk))
if mism:
    print("\n不一致明细(前25条):")
    for code, sd, bad, dd in mism[:25]:
        print("  %s %s: %s" % (code, sd, " | ".join("%s mine=%s orig=%s" % (k, dd[k][0], dd[k][1]) for k in bad)))
json.dump({"n_cmp": n_cmp, "n_ok": n_ok,
           "mismatch": [{"code": c, "date": s, "bad": b, "detail": {k: list(vv) for k, vv in d.items()}} for c, s, b, d in mism]},
          open(os.path.join(HERE, "bp1_verify2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved -> bp1_verify2.json")
