# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step1 特征预处理(逐行复现 niigu_judge.py + shunshi_buy_points.py)
bp1 = 买点一(平台突破买点):
  L0 N类门控: n_raw_score >= 40 (n_raw = int((dim1+dim2+dim3+dim4+dim6)*stage_mult), 不乘大盘折扣)
  L1 大势门: N>=25即过(L0已保证>=40, 自动通过)
  L2 小势门: 只降confidence, 不否决信号
  技术五条件:
    (1) K线>=60根
    (2) 平台振幅: HHV(H,20)/LLV(L,20) 近20日不含今日 < 25%
    (3) 波动率压缩: ATR(14,Wilder)/MA(C,14) < 0.06
    (4) 突破: C > HHV(H,20不含今日) 且 C/HHV > 1.03
    (5) 量能: V > MA5(V不含今日)*1.5 且 V > MA60(V不含今日)*1.5
口径: T日信号收盘后可知 -> T+1开盘入场 -> T+20收盘出场
宇宙: (DB codes ∩ stock_list_all.json) − 指数
RPS参考系可配: all=全宇宙; pool=池内1242(校准模式)
注意: 所有滚动指标按"行号"回看(与原引擎一致, 停牌日不占位)
"""
import sqlite3, os, pickle, json, time, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
OUT = os.path.join(HERE, "bp1_features.pkl")
INDEX_CODES = {"000001", "000905", "000852"}
MIN_KLINES = 60

RPS_REF_MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
OUT_ARG = sys.argv[2] if len(sys.argv) > 2 else OUT
POOL_FILE = sys.argv[3] if len(sys.argv) > 3 else POOL

t0 = time.time()

# ---------------- 工具 ----------------
def ma_arr(arr, period):
    # 滑动窗口mean与np.mean逐位一致(实测p=5/10/14/20/60全一致); cumsum版有~1e-14噪声,
    # 会在百分位tie与阈值比较上造成边界翻转(000571 d4实例)
    n = len(arr); r = np.full(n, np.nan)
    if n < period: return r
    r[period-1:] = np.lib.stride_tricks.sliding_window_view(arr, period).mean(axis=1)
    return r

def expmean_until(arr, upto):
    """mean(arr[:upto]) 序列, upto>=1"""
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    return cs[upto]/upto

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
    """r[t] = corr(x[t-w+1:t+1], y[t-w+1:t+1]) (w个点, 截至t) — 与np.corrcoef(diff)一致"""
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
    """对每个t: 100*mean(series[max(0,t-w+1):t+1]中<=series[t]的比例, 剔NaN); 有效值<10 或 series[t]为NaN -> NaN
    与原引擎_dim_volatility一致: lookback=min(120,n_truncated) -> 短历史用扩展窗口(前补NaN实现);
    min_n: 原引擎的n>=X门禁(BBW 30 / ATR 24), 截至t不足则NaN"""
    n = len(series); r = np.full(n, np.nan)
    if n < 10: return r
    pad = np.concatenate([np.full(w-1, np.nan), series])
    sv = np.lib.stride_tricks.sliding_window_view(pad, w)   # sv[t]=series[max(0,t-w+1):t+1], 前补NaN
    cur = series
    valid = ~np.isnan(sv)
    cnt = valid.sum(axis=1)
    le = ((sv <= cur[:, None]) & valid).sum(axis=1)
    ok = (cnt >= 10) & (~np.isnan(cur))
    r[:] = np.where(ok, le/np.where(ok, cnt, 1)*100, np.nan)
    if min_n is not None:
        r[:min_n-1] = np.nan
    return r

# ---------------- 数据加载 ----------------
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
stock_list = json.load(open(os.path.join(SRC, "stock_list_all.json"), encoding="utf-8"))
list_codes = set(s["code"] if isinstance(s, dict) else s for s in stock_list)
db_codes = [r[0] for r in cur.execute("SELECT DISTINCT code FROM klines ORDER BY code")]
universe = [c for c in db_codes if c in list_codes and c not in INDEX_CODES]
print("DB codes: %d | canonical universe: %d" % (len(db_codes), len(universe)))

if RPS_REF_MODE == "pool":
    pool_codes = set(s["code"] for s in json.load(open(POOL_FILE, encoding="utf-8"))["stocks"])
    rps_ref = [c for c in universe if c in pool_codes]
else:
    rps_ref = list(universe)
rps_ref_set = set(rps_ref)
print("RPS参考系: %s (%d只)" % (RPS_REF_MODE, len(rps_ref)))

raw = {}
for code in universe:
    rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date", (code,)).fetchall()
    if len(rows) < MIN_KLINES: continue
    raw[code] = rows
bench = {}
for ic in INDEX_CODES:
    brows = cur.execute("SELECT date,close FROM klines WHERE code=? ORDER BY date", (ic,)).fetchall()
    bench[ic] = {"date": [r[0] for r in brows], "close": np.array([float(r[1]) for r in brows], dtype=np.float32)}
con.close()
print("有效K线(>=60行): %d 只, %.0fs" % (len(raw), time.time()-t0))

# ---------------- 逐票指标 ----------------
feats = {}
for idx, (code, rows) in enumerate(raw.items()):
    dates = np.array([r[0] for r in rows])
    o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
    l = np.array([float(r[3]) for r in rows]); c = np.array([float(r[4]) for r in rows])
    v = np.array([float(r[5]) for r in rows]); n = len(c)
    ma5 = ma_arr(c, 5); ma10 = ma_arr(c, 10); ma20 = ma_arr(c, 20); ma60 = ma_arr(c, 60)
    vma5_1 = np.concatenate([[np.nan], ma_arr(v, 5)[:-1]])       # mean(v[t-5:t]) 不含今日
    vma60_1 = np.full(n, np.nan)                                  # mean(v[t-60:t]) 或 mean(v[:t])
    for t in range(1, n):
        vma60_1[t] = (v[t-60:t].mean() if t >= 61 else v[:t].mean())
    vma20 = ma_arr(v, 20)
    vr5_today = v/np.where(ma_arr(v, 5) > 0, ma_arr(v, 5), np.nan)   # dim2A用含今日
    atr14 = atr_wilder(h, l, c, 14)          # bp1技术条件3用(Wilder, 与shunshi引擎一致)
    tr = np.full(n, np.nan)                   # d4用简单MA(TR,14) — 与niigu引擎一致(tr[0]=NaN, ATR第13位无效)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr_simple = ma_arr(tr, 14)              # N类波动率维度用
    ma_c14 = ma_arr(c, 14)
    std20 = np.full(n, np.nan)
    if n >= 20:
        std20[19:] = np.lib.stride_tricks.sliding_window_view(c, 20).std(axis=1)
    bbw = np.full(n, np.nan)
    okb = (~np.isnan(ma20)) & (~np.isnan(std20)) & (ma20 > 0)
    # 与原引擎逐位一致: upper=sma+2std, lower=sma-2std, bbw=(upper-lower)/sma (非4*std/sma, 浮点路径不同)
    upper = ma20 + 2*std20; lower = ma20 - 2*std20
    bbw[okb] = (upper[okb]-lower[okb])/ma20[okb]
    atr_ratio = np.full(n, np.nan)
    ok2 = (~np.isnan(atr_simple)) & (c > 0)
    atr_ratio[ok2] = atr_simple[ok2]/c[ok2]
    bbw_pct = rolling_pct_of_last(bbw, 120, min_n=30)     # 原引擎: n>=30 才算BBW分位
    atr_pct = rolling_pct_of_last(atr_ratio, 120, min_n=24)  # 原引擎: n>=atr_period+10=24
    # ---- dim1 趋势结构 ----
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
    # ---- dim2 量价关系 ----
    d2 = np.zeros(n)
    shrink = (v < vma20*0.6) & (~np.isnan(vma20))
    has_shrink = rolling_any(shrink, 20)
    d2 += np.where(has_shrink & (vr5_today > 1.5), 10, np.where(has_shrink & (vr5_today > 1.2), 7, np.where(has_shrink, 4, 0)))
    dv = np.diff(v, prepend=np.nan); dc = np.diff(c, prepend=np.nan)
    vp_corr = win_corr_upto(dv, dc, 5)
    d2 += np.where(vp_corr > 0.5, 5, np.where(vp_corr > 0, 3, 0))
    # C. 底部放量(向量化, 按offset顺序)
    bullish = c > o
    lvr = v/np.where(vma20 > 0, vma20, np.nan)
    is_b5 = bullish & (lvr > 2.0) & (~np.isnan(lvr))
    is_b15 = bullish & (lvr > 1.5) & (~np.isnan(lvr))
    bottom_score = np.zeros(n)
    if n >= 60:
        am = np.lib.stride_tricks.sliding_window_view(l, 60).argmin(axis=1)  # 窗口[0..59]内argmin
        gidx = np.arange(59, n)-59+am                                   # 全局最低价位置
        gidx_c = np.clip(gidx, 0, n-1)
        found = np.zeros(n-59, bool); score = np.zeros(n-59)
        jend = np.arange(59, n)                              # 各窗口末日索引
        for off in (-1, 0, 1, 2, 3):
            # 上锚=窗口末日j而非n-1: 原引擎在"截断到t"的序列上运行, offset越界即跳过;
            # 若锚到n-1, gidx+off>t 时会读到t之后的K线 = 前视偏差(603353 20240603 实例)
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
    cross[1:] = up[1:] & (~up[:-1]) & valid_prev[1:]      # 上穿需前后行均有效(原引擎None检查语义)
    last_cross = np.maximum.accumulate(np.where(cross, np.arange(n), -1))
    trend_age = np.where(last_cross >= 0, np.arange(n)-last_cross, np.nan)   # 截断口径: i - 最后一次上穿位置
    age_mult = np.ones(n)
    age_mult[up & (~np.isnan(trend_age)) & (trend_age >= 120)] = 0.85
    age_mult[up & (~np.isnan(trend_age)) & (trend_age >= 60) & (trend_age < 120)] = 0.90
    stage_mult = np.minimum(ext_mult, age_mult)
    # ---- bp1 技术五条件 ----
    cond_len = np.zeros(n, bool); cond_len[MIN_KLINES-1:] = True
    ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    if n >= 21:
        ph[20:] = np.lib.stride_tricks.sliding_window_view(h, 20).max(axis=1)[:-1]   # max(h[t-20:t]) 不含今日
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
    # ---- L2(描述统计) ----
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
    # ---- 前瞻/后视 ----
    fwd20 = np.full(n, np.nan)
    if n > 20: fwd20[:n-20] = c[20:]/c[:n-20]-1.0
    t1_20 = np.full(n, np.nan)
    if n > 20: t1_20[:n-20] = c[20:n]/o[1:n-19]-1.0
    pre20 = np.full(n, np.nan); pre20[20:] = c[20:]/c[:-20]-1.0
    pre60 = np.full(n, np.nan); pre60[60:] = c[60:]/c[:-60]-1.0
    ret1 = np.full(n, np.nan); ret1[1:] = c[1:]/c[:-1]-1.0
    vol20 = np.full(n, np.nan)
    if n >= 20:
        vol20[19:] = np.lib.stride_tricks.sliding_window_view(ret1, 20).std(axis=1)
    maxh20 = np.full(n, np.nan)
    if n >= 20:
        maxh20[19:] = np.lib.stride_tricks.sliding_window_view(h, 20).max(axis=1)
    dd20 = (l-maxh20)/np.where(maxh20 > 0, maxh20, np.nan)
    mae20 = np.full(n, np.nan); mfe20 = np.full(n, np.nan)
    below_days = np.full(n, np.nan); first5_min = np.full(n, np.nan); last15_max = np.full(n, np.nan)
    for i in range(n-1):
        e = o[i+1]
        if not (e > 0): continue
        seg_lo = l[i+1:min(i+21, n)]; seg_hi = h[i+1:min(i+21, n)]; seg_c = c[i+1:min(i+21, n)]
        if len(seg_c) < 20: continue
        mae20[i] = seg_lo.min()/e-1.0
        mfe20[i] = seg_hi.max()/e-1.0
        below_days[i] = (seg_c < e).sum()
        first5_min[i] = seg_c[:5].min()/e-1.0
        last15_max[i] = seg_c[5:].max()/e-1.0
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n-1):
        pre = v[i-20:i]
        post = v[i+1:min(i+21, n)]
        if len(post) >= 5 and pre.mean() > 0:
            vol_ratio[i] = post.mean()/pre.mean()
    feats[code] = {
        "date": dates, "rows": rows,
        "o": o.astype(np.float32), "h": h.astype(np.float32), "l": l.astype(np.float32),
        "c": c.astype(np.float32), "v": v.astype(np.float32),
        "bp1_tech": bp1_tech, "bp1_sig": np.zeros(n, bool),
        "d1": d1.astype(np.float32), "d2": d2.astype(np.float32),
        "d4": d4.astype(np.float32), "d6": d6.astype(np.float32), "d3_extra": d3_extra.astype(np.float32),
        "stage_mult": stage_mult.astype(np.float32), "ext": ext.astype(np.float32),
        "stage_mult64": stage_mult,   # float64原值: n_raw=floor(sum*mult)须用float64(见RPS段注释)
        "trend_age": trend_age.astype(np.float32), "l2_pass": l2_pass,
        "fwd20": fwd20.astype(np.float32), "t1_20": t1_20.astype(np.float32),
        "pre20": pre20.astype(np.float32), "pre60": pre60.astype(np.float32),
        "dd20": dd20.astype(np.float32), "vol20": vol20.astype(np.float32),
        "mae20": mae20.astype(np.float32), "mfe20": mfe20.astype(np.float32),
        "below_days": below_days.astype(np.float32),
        "first5_min": first5_min.astype(np.float32), "last15_max": last15_max.astype(np.float32),
        "vol_ratio": vol_ratio.astype(np.float32),
        "breakout_amp": (brk*100).astype(np.float32), "platform_range": range_pct.astype(np.float32),
        "vol_atr_ratio": vr_atr.astype(np.float32), "vr5_1": (v/np.where(vma5_1 > 0, vma5_1, np.nan)).astype(np.float32),
        "vr_long": (v/np.where(vma60_1 > 0, vma60_1, np.nan)).astype(np.float32),
        "vr5_today": vr5_today.astype(np.float32), "vp_corr": vp_corr.astype(np.float32),
        "rsi": rsi.astype(np.float32), "ma20_slope": ma20_slope.astype(np.float32),
    }
    if (idx+1) % 500 == 0:
        print("  %d/%d  %.0fs" % (idx+1, len(raw), time.time()-t0))
print("逐票指标完成 %.0fs, RPS截面计算..." % (time.time()-t0))

# ---------------- RPS 截面(行号口径) ----------------
all_dates = sorted({r[0] for rows_ in raw.values() for r in rows_})
dmap = {d: i for i, d in enumerate(all_dates)}
nd = len(all_dates)
codes = list(feats.keys())
nc = len(codes)
Cm = np.full((nc, nd), np.nan)
RowOf = np.full((nc, nd), -1, dtype=np.int32)   # 每票每轴位置的行号(-1=无数据)
dateidx_of_row = []                              # 每票每行的轴索引
for ci, code in enumerate(codes):
    rows = raw[code]
    dr = np.empty(len(rows), dtype=np.int32)
    for ri, r in enumerate(rows):
        k = dmap[r[0]]
        Cm[ci, k] = float(r[4])
        RowOf[ci, k] = ri
        dr[ri] = k
    dateidx_of_row.append(dr)
print("日期轴 %d 天 (%s~%s)" % (nd, all_dates[0], all_dates[-1]))

ref_mask = np.array([code in rps_ref_set for code in codes])
ref_ids = np.where(ref_mask)[0]
n_ref = len(ref_ids)
Cm_ref = Cm[ref_mask]
RowOf_ref = RowOf[ref_mask]
dateidx_ref = [dateidx_of_row[i] for i in ref_ids]
start_ref = np.array([0]*n_ref)  # 全部从行0开始

# 预计算每票的 close-by-row  padded 矩阵(行号口径), 供向量化 gather
max_rows = max((len(dr) for dr in dateidx_ref), default=1)
Cbrow_mat = np.full((n_ref, max_rows), np.nan)
for jj in range(n_ref):
    dr = dateidx_ref[jj]
    Cbrow_mat[jj, :len(dr)] = Cm_ref[jj, dr]

rps_mat = np.full((nc, nd), np.nan)
for k in range(nd):
    rowk = RowOf_ref[:, k]                       # 每票在k日的行号(-1无数据)
    valid = rowk >= 0
    nrows = rowk-start_ref+1                       # 截至k的行数
    if int(((nrows >= 20) & valid).sum()) < 10: continue   # 原引擎门: all_closes_map(>=20根的票)>=10 才进RPS模式
    wk = np.minimum(250, nrows).astype(np.int32)
    wk[wk < 1] = 1
    for w in np.unique(wk[valid]):
        # 原引擎口径: rps_period=min(250,当前票n), 截面=所有 len>=rps_period 的票(不是仅==);
        # 收益一律按当前票的 rps_period 计算。年轻票(n<250)的截面含成熟票。
        mask = valid & (wk >= w)
        m_ids = np.where(mask)[0]
        if len(m_ids) < 10: continue
        # 目标行 = rowk - w + 1 (>=start)
        trow = rowk[m_ids]-w+1
        okrow = trow >= start_ref[m_ids]
        trow_c = np.clip(trow, 0, max_rows-1)
        base = Cbrow_mat[m_ids, trow_c]                     # 向量化gather: 每票w个交易日前的收盘
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = np.where(okrow, Cm_ref[m_ids, k]/base-1.0, np.nan)
        rr = rets[~np.isnan(rets)]
        if len(rr) < 1: continue                  # 原引擎: if all_returns(非空即算)
        srt = np.sort(rr)
        my = rets[~np.isnan(rets)]
        ranks = np.searchsorted(srt, my, side="right")/len(rr)*100
        good = m_ids[~np.isnan(rets)]
        rps_mat[ref_ids[good], k] = ranks
    if (k+1) % 200 == 0:
        print("  RPS %d/%d  %.0fs" % (k+1, nd, time.time()-t0))

for ci, code in enumerate(codes):
    ft = feats[code]
    drow = dateidx_of_row[ci]                 # 每行的日期轴索引
    rps = rps_mat[ci, drow]                   # 行号口径的RPS
    d3 = np.where(rps >= 90, 20, np.where(rps >= 80, 16, np.where(rps >= 70, 12,
         np.where(rps >= 50, 8, np.where(rps >= 30, 4, np.where(~np.isnan(rps), 0, 0))))))
    d3 = np.minimum(d3+ft["d3_extra"], 20)
    # float64 计算: 原引擎 dims为Python int, stage_mult为Python float(0.90等字面量);
    # 若用float32的stage_mult, 50*0.9=44.999998 -> floor 44 而非 45(002463等边界翻车)
    s64 = (ft["d1"].astype(np.float64)+ft["d2"].astype(np.float64)+d3.astype(np.float64)
           +ft["d4"].astype(np.float64)+ft["d6"].astype(np.float64))
    n_raw = np.floor(s64*ft["stage_mult64"]).astype(np.int32)
    ft["rps"] = rps.astype(np.float32)
    ft["d3"] = d3.astype(np.float32)
    ft["n_raw"] = n_raw
    ft["l0_pass"] = n_raw >= 40
    ft["bp1_sig"] = ft["bp1_tech"] & ft["l0_pass"]

# 清理大字段
for code in feats:
    feats[code].pop("rows", None)
    feats[code].pop("stage_mult64", None)

with open(OUT_ARG, "wb") as f:
    pickle.dump({"feats": feats, "bench": bench}, f, protocol=4)
nsig = sum(int(ft["bp1_sig"].sum()) for ft in feats.values())
print("done: %d stocks, bp1信号 %d 条, %.1f MB, %.0fs -> %s" % (
    len(feats), nsig, os.path.getsize(OUT_ARG)/1e6, time.time()-t0, OUT_ARG))
