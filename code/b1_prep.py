# -*- coding: utf-8 -*-
"""
B1因子重构评估 — Step1 特征预处理
从 kline_cache_all.db 加载K线, 计算 B1 依赖指标与检验特征, 缓存 b1_features.pkl

口径约定(重构版统一口径):
  T日     = 信号日(收盘后可知, 当日最低价已发生)
  T+1     = 开盘入场
  T+20    = 收盘出场(持有20个交易日)
  fwd_c2c = 纯信号层参考收益 close[T+k]/close[T]-1

宇宙定义( canonical universe ):
  (DB codes ∩ stock_list_all.json) − {000001}   # 000001槽位被上证指数占用
  = 5016 只; 剔除 3 只指数(000001/000905/000852)与 23 只名单外代码(含ETF)
"""
import sqlite3, os, pickle, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
DB = os.path.join(SRC, "kline_cache_all.db")
OUT = os.path.join(HERE, "b1_features.pkl")
INDEX_CODES = {"000001", "000905", "000852"}

t0 = time.time()

# ---------------- 指标计算(与 backtest_b1_all_vec.py / 生产引擎逐行一致) ----------------
def ma_cumsum(arr, period):
    n = len(arr); result = np.full(n, np.nan)
    if n < period: return result
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    result[period-1:] = (cs[period:] - cs[:-period]) / period
    return result

def ema_series(arr, period):
    n = len(arr); result = np.full(n, np.nan)
    if n < period: return result
    alpha = 2.0 / (period + 1)
    result[period-1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def calc_indicators(closes, highs, lows):
    """返回 KDJ_J, 知行趋势线, 知行多空线 — 与生产 calc_zge_indicators 一致"""
    n = len(closes)
    low9 = np.full(n, np.nan); high9 = np.full(n, np.nan)
    for i in range(8, n):
        low9[i] = np.min(lows[i-8:i+1]); high9[i] = np.max(highs[i-8:i+1])
    denom = high9 - low9
    rsv = np.where(denom > 0, (closes - low9) / denom * 100, 50.0)
    k = np.full(n, 50.0); d = np.full(n, 50.0)
    for i in range(9, n):
        k[i] = 2/3 * k[i-1] + 1/3 * rsv[i]
        d[i] = 2/3 * d[i-1] + 1/3 * k[i]
    j = 3*k - 2*d
    ema10 = ema_series(closes, 10)
    ema10_valid = ema10[~np.isnan(ema10)]
    zx_t = ema_series(ema10_valid, 10)
    zx_trend = np.full(n, np.nan)
    zx_t_valid = zx_t[~np.isnan(zx_t)]
    valid_start = 18
    if len(zx_t_valid) > 0:
        zx_trend[valid_start:valid_start+len(zx_t_valid)] = zx_t_valid
    zx_dk = (ma_cumsum(closes, 14) + ma_cumsum(closes, 28) +
             ma_cumsum(closes, 57) + ma_cumsum(closes, 114)) / 4
    return j, zx_trend, zx_dk

def fwd_return(closes, k):
    """close[i+k]/close[i]-1, 尾部k个置NaN"""
    n = len(closes); r = np.full(n, np.nan)
    if n > k: r[:n-k] = closes[k:]/closes[:n-k] - 1.0
    return r

def t1_return(o, c, k):
    """T+1开盘入场, T+k收盘出场(与原引擎 run_one 一致): c[i+k]/o[i+1]-1"""
    n = len(o); r = np.full(n, np.nan)
    if n > k: r[:n-k] = c[k:n]/o[1:n-k+1] - 1.0
    return r

def rolling_max(a, w):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(a); r = np.full(n, np.nan)
    if n >= w:
        r[w-1:] = sliding_window_view(a, w).max(axis=1)
    return r

def rolling_std(a, w):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(a); r = np.full(n, np.nan)
    if n >= w:
        r[w-1:] = sliding_window_view(a, w).std(axis=1)
    return r

def rolling_mean(a, w):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(a); r = np.full(n, np.nan)
    if n >= w:
        r[w-1:] = sliding_window_view(a, w).mean(axis=1)
    return r

def post_window_stats(v, lo, hi, o1, w_pre=20, w_post=20):
    """量能比 mean(v[i+1:i+21])/mean(v[i-20:i]) — 跟风代理"""
    n = len(v); r = np.full(n, np.nan)
    for i in range(w_pre, n-1):
        pre = v[i-w_pre:i]
        post = v[i+1:min(i+1+w_post, n)]
        if len(post) >= 5 and pre.mean() > 0:
            r[i] = post.mean()/pre.mean()
    return r

# ---------------- 主流程 ----------------
con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
cur = con.cursor()
stock_list = json.load(open(os.path.join(SRC, "stock_list_all.json"), encoding="utf-8"))
list_codes = set(s["code"] if isinstance(s, dict) else s for s in stock_list)
db_codes = [r[0] for r in cur.execute("SELECT DISTINCT code FROM klines ORDER BY code")]
universe = [c for c in db_codes if c in list_codes and c not in INDEX_CODES]
print("DB codes: %d | list: %d | canonical universe: %d" % (len(db_codes), len(list_codes), len(universe)))

feats = {}
for idx, code in enumerate(universe):
    rows = cur.execute("SELECT date,open,high,low,close,volume FROM klines WHERE code=? ORDER BY date", (code,)).fetchall()
    if len(rows) < 130:   # 指标最短依赖(MA114) + 少量余量
        continue
    dates = np.array([r[0] for r in rows])
    o = np.array([float(r[1]) for r in rows]); h = np.array([float(r[2]) for r in rows])
    l = np.array([float(r[3]) for r in rows]); c = np.array([float(r[4]) for r in rows])
    v = np.array([float(r[5]) for r in rows])
    n = len(c)
    j, zt, zd = calc_indicators(o*0+c, h, l)  # closes=c
    atr = 0.0214 * c
    # ---- 信号与条件分解 ----
    cond_j = np.where(np.isnan(j), 0, (j < 20)).astype(np.int8)
    touch_zt = (l <= zt*1.001) & (l >= (zt - 0.5*atr)*0.995) & ~np.isnan(zt)
    touch_zd = (l <= zd*1.001) & (l >= (zd - 0.5*atr)*0.995) & ~np.isnan(zd)
    hold_zt = touch_zt & (c >= zt*0.99)
    hold_zd = touch_zd & (c >= zd*0.99)
    sig_dc = (cond_j.astype(bool) & hold_zt).astype(np.int8)
    sig_dk = (cond_j.astype(bool) & hold_zd).astype(np.int8)
    sig_base = ((sig_dc | sig_dk) > 0).astype(np.int8)
    # 触线但不收稳(被洗破? 实际是"破位") — 用于检验"不破"过滤器的价值
    break_zt = (touch_zt & (c < zt*0.99)).astype(np.int8)
    # ---- 前瞻/后视收益 ----
    fwd19 = fwd_return(c, 19)     # 旧CSV口径 close[T+19]/close[T]
    fwd20 = fwd_return(c, 20)     # close[T+20]/close[T]
    t1_20 = t1_return(o, c, 20)   # T+1开盘→T+20收盘(无止损)
    pre20 = np.full(n, np.nan); pre20[20:] = c[20:]/c[:-20] - 1.0
    pre60 = np.full(n, np.nan); pre60[60:] = c[60:]/c[:-60] - 1.0
    # ---- 趋势强度/回撤/波动 ----
    maxh20 = rolling_max(h, 20)
    dd20 = (l - maxh20) / maxh20                    # 当日最低相对20日高回撤
    ret1 = np.full(n, np.nan); ret1[1:] = c[1:]/c[:-1] - 1.0
    vol20 = rolling_std(ret1, 20)
    dist_zt = (c - zt) / zt
    dist_zd = (c - zd) / zd
    depth_zt = np.where(touch_zt, (zt - l) / zt, np.nan)   # 触线深度(最低价穿线多少)
    depth_zd = np.where(touch_zd, (zd - l) / zd, np.nan)
    ma114 = ma_cumsum(c, 114)
    ma60 = ma_cumsum(c, 60)
    # ---- 路径特征(基于T+1开盘入场) ----
    n_ = n
    mae20 = np.full(n_, np.nan); mfe20 = np.full(n_, np.nan)
    below_days = np.full(n_, np.nan); first5_min = np.full(n_, np.nan); last15_max = np.full(n_, np.nan)
    for i in range(n_-1):
        e = o[i+1]
        if not (e > 0): continue
        seg_lo = l[i+1:min(i+21, n_)]; seg_hi = h[i+1:min(i+21, n_)]; seg_c = c[i+1:min(i+21, n_)]
        if len(seg_c) < 20: continue
        mae20[i] = seg_lo.min()/e - 1.0
        mfe20[i] = seg_hi.max()/e - 1.0
        below_days[i] = (seg_c < e).sum()
        first5_min[i] = seg_c[:5].min()/e - 1.0
        last15_max[i] = seg_c[5:].max()/e - 1.0
    vol_ratio = post_window_stats(v, l, h, o)
    feats[code] = {
        "date": dates,
        "o": o.astype(np.float32), "h": h.astype(np.float32), "l": l.astype(np.float32),
        "c": c.astype(np.float32), "v": v.astype(np.float32),
        "j": j.astype(np.float32), "zt": zt.astype(np.float32), "zd": zd.astype(np.float32),
        "sig_base": sig_base, "sig_dc": sig_dc, "sig_dk": sig_dk,
        "cond_j": cond_j, "touch_zt": touch_zt.astype(np.int8), "touch_zd": touch_zd.astype(np.int8),
        "hold_zt": hold_zt.astype(np.int8), "break_zt": break_zt,
        "fwd19": fwd19.astype(np.float32), "fwd20": fwd20.astype(np.float32),
        "t1_20": t1_20.astype(np.float32),
        "pre20": pre20.astype(np.float32), "pre60": pre60.astype(np.float32),
        "dd20": dd20.astype(np.float32), "vol20": vol20.astype(np.float32),
        "dist_zt": dist_zt.astype(np.float32), "dist_zd": dist_zd.astype(np.float32),
        "depth_zt": depth_zt.astype(np.float32), "depth_zd": depth_zd.astype(np.float32),
        "ma114": ma114.astype(np.float32), "ma60": ma60.astype(np.float32),
        "mae20": mae20.astype(np.float32), "mfe20": mfe20.astype(np.float32),
        "below_days": below_days.astype(np.float32),
        "first5_min": first5_min.astype(np.float32), "last15_max": last15_max.astype(np.float32),
        "vol_ratio": vol_ratio.astype(np.float32),
    }
    if (idx+1) % 500 == 0:
        print("  %d/%d  %.0fs" % (idx+1, len(universe), time.time()-t0))

# 指数基准序列(单独存)
bench = {}
for ic in INDEX_CODES:
    rows = cur.execute("SELECT date,close FROM klines WHERE code=? ORDER BY date", (ic,)).fetchall()
    bench[ic] = {"date": [r[0] for r in rows], "close": np.array([float(r[1]) for r in rows], dtype=np.float32)}
con.close()

with open(OUT, "wb") as f:
    pickle.dump({"feats": feats, "bench": bench}, f, protocol=4)
sz = os.path.getsize(OUT)/1e6
print("done: %d stocks, bench %s, %.1f MB, %.0fs" % (len(feats), {k: len(v["date"]) for k, v in bench.items()}, sz, time.time()-t0))
