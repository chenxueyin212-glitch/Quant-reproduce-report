# -*- coding: utf-8 -*-
"""
六信号因子体检补全 — 统一补上标准因子评估缺失指标
模块A: Pearson IC + RankIC（全截面二元信号因子，持有期 5/10/20 衰减曲线）
模块B: 风格暴露（OHLCV 可得代理：ln成交额/beta60/mom60/vol20/rev20）+ 风格调整后 alpha
       —— 真 Barra 需市值+行业+基本面，数据源不含（D12 属实），以代理风格如实登记
模块C: 换手率（持仓日度换手/持仓重叠率/信号粘性）
模块D: 自相关性（策略日收益 lag-1 自相关）
口径: T日信号收盘后可知 → T+1开盘入场 → T+h收盘（h=5/10/20），训练<20250908
输出: healthcheck_results.json
"""
import pickle, os, json, math
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_END = "20250908"
D0, D1 = "20231101", "20260821"
PCT = 100.0
ROUND_TRIP = (0.00025*2 + 0.0005 + 0.001*2)*PCT

SIGNALS = [
    ("B1(倒车/多空)", "b1_features.pkl", "sig_base"),
    ("B2", "b2_features.pkl", "b2_sig"),
    ("bp1(突破追入)", "bp1_features.pkl", "bp1_sig"),
    ("补票(单针下20)", "bupiao_features.pkl", "sig"),
    ("搬砖(多空临界)", "banzhuan_features.pkl", "sig"),
    ("bp2(均线回踩)", "bp2_features.pkl", "sig"),
]

def zscore_cs(v):
    """当日截面 z 分（nan 跳过）"""
    m = np.nanmean(v); s = np.nanstd(v)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(v)
    return (v - m) / s

def horizons(o, c, n):
    """r_h[t] = c[t+h]/o[t+1]-1, h=5/10/20"""
    out = {}
    for h in (5, 10, 20):
        r = np.full(n, np.nan)
        if n > h + 1:
            r[:n-h] = (c[h:] / o[1:n-h+1]) - 1.0   # r[t] = c[t+h]/o[t+1]-1, t=0..n-h-1
        out[h] = r
    return out

def styles_of(code, ft, mkt_cache, raw_src=None):
    """计算/取缓存的风格代理（按 code 缓存；同 code 阵列全库一致）。
    bupiao/banzhuan/bp2 特征文件不含原始 OHLCV——从 raw_src（bp1_features）回退取数，
    三者均由 bp1_features 派生，阵列逐位对齐。"""
    if code in STYLE_CACHE:
        return STYLE_CACHE[code]
    if "o" in ft:
        src = ft
    else:
        src = (raw_src or {}).get(code)
        if src is None or len(src["o"]) != len(ft["date"]):
            return None
    o = src["o"].astype(np.float64); h = src["h"].astype(np.float64)
    l = src["l"].astype(np.float64); c = src["c"].astype(np.float64)
    v = src["v"].astype(np.float64)
    n = len(c)
    amt = c * v
    # ln 20日均成交额（规模/流动性代理）
    ln_amt = np.full(n, np.nan)
    if n >= 20:
        w = np.lib.stride_tricks.sliding_window_view(amt, 20).mean(axis=1)
        ln_amt[19:] = np.log(np.maximum(w, 1.0))
    # 日收益
    ret = np.full(n, np.nan)
    ret[1:] = c[1:]/c[:-1] - 1.0
    # beta60 vs 市场（等权）—— 矢量化滚动 cov/var（NaN 跳过按有效计数）
    mret = mkt_cache["mret"]; mdidx = mkt_cache["mdidx"]
    dates = np.array([str(x).replace("-", "") for x in ft["date"]])
    mkt = np.full(n, np.nan)
    for i in range(n):
        j = mdidx.get(dates[i], -1)
        if j >= 0:
            mkt[i] = mret[j]
    beta = np.full(n, np.nan)
    W = 60
    if n >= W:
        both = (~np.isnan(ret)) & (~np.isnan(mkt))
        r0 = np.where(both, ret, 0.0); m0 = np.where(both, mkt, 0.0)
        rm = r0 * m0; r2 = r0 * r0; m2 = m0 * m0
        okf = both.astype(np.float64)
        sv_r = np.lib.stride_tricks.sliding_window_view(r0, W).sum(axis=1)
        sv_m = np.lib.stride_tricks.sliding_window_view(m0, W).sum(axis=1)
        sv_rm = np.lib.stride_tricks.sliding_window_view(rm, W).sum(axis=1)
        sv_r2 = np.lib.stride_tricks.sliding_window_view(r2, W).sum(axis=1)
        sv_m2 = np.lib.stride_tricks.sliding_window_view(m2, W).sum(axis=1)
        sv_ok = np.lib.stride_tricks.sliding_window_view(okf, W).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sv_rm/W - (sv_r/W)*(sv_m/W)
            var_m = sv_m2/W - (sv_m/W)**2
            b = np.where((sv_ok >= 40) & (var_m > 0), cov/np.maximum(var_m, 1e-18), np.nan)
        beta[W-1:] = b
    # mom60 / vol20 / rev20
    mom60 = np.full(n, np.nan)
    if n > 60:
        mom60[60:] = (c[60:] / c[:n-60] - 1.0) * 100.0
    vol20 = np.full(n, np.nan)
    if n >= 21:
        rr = np.zeros(n); rr[1:] = ret[1:]
        rr = np.where(np.isnan(rr), 0.0, rr)
        okf = (~np.isnan(ret)).astype(np.float64); okf[0] = 0.0
        W2 = 20
        s1 = np.lib.stride_tricks.sliding_window_view(rr, W2).sum(axis=1)
        s2 = np.lib.stride_tricks.sliding_window_view(rr*rr, W2).sum(axis=1)
        sc = np.lib.stride_tricks.sliding_window_view(okf, W2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(sc >= 15, s2/np.maximum(sc, 1) - (s1/np.maximum(sc, 1))**2, np.nan)
        vol20[W2-1:] = np.sqrt(np.maximum(var, 0.0)) * 100.0
    rev20 = np.full(n, np.nan)
    if n > 20:
        rev20[20:] = (c[20:] / c[1:n-19] - 1.0) * 100.0  # 19期口径与系列一致
    hz = horizons(o, c, n)
    st = {"ln_amt": ln_amt.astype(np.float32), "beta": beta.astype(np.float32),
          "mom60": mom60.astype(np.float32), "vol20": vol20.astype(np.float32),
          "rev20": rev20.astype(np.float32),
          "r5": hz[5].astype(np.float32), "r10": hz[10].astype(np.float32),
          "r20": hz[20].astype(np.float32), "dates": dates, "c": c}
    STYLE_CACHE[code] = st
    return st

STYLE_CACHE = {}

def period_of(ds): return "训练期" if ds < TRAIN_END else "验证期"

def desc(v):
    v = np.array(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return {"n": int(len(v)), "mean": float(v.mean()),
            "win": float(100*(v > 0).mean()), "std": float(v.std())}

def ic_block(ic_list):
    arr = np.array([x for x in ic_list if x is not None and np.isfinite(x)])
    if len(arr) == 0: return None
    return {"n_days": int(len(arr)), "mean": float(arr.mean()),
            "icir": float(arr.mean()/arr.std()) if arr.std() > 0 else None,
            "pos_ratio": float(100*(arr > 0).mean()),
            "t": float(arr.mean()/(arr.std()/np.sqrt(len(arr)))) if arr.std() > 0 else None}

R = {"meta": {"styles": ["ln_amt20(规模/流动性代理)", "beta60(市场β)",
                         "mom60(动量)", "vol20(波动)", "rev20(反转)"],
              "barra_note": "真 Barra 风格需市值/行业/基本面，数据源不含（D12）——本体检用 OHLCV 代理风格，"
                            "行业暴露无法检验，登记为 D41"}}

# ============ 市场收益（等权全A） ============
print("构建市场基准...")
BP1_RAW = pickle.load(open(os.path.join(HERE, "bp1_features.pkl"), "rb"))["feats"]
ref = BP1_RAW
all_dates = sorted({str(ds).replace("-", "") for ft in ref.values() for ds in ft["date"]})
didx_all = {dd: i for i, dd in enumerate(all_dates)}
tot = np.zeros(len(all_dates)); cnt = np.zeros(len(all_dates))
for code, ft in ref.items():
    c = ft["c"].astype(np.float64)
    dates = np.array([str(x).replace("-", "") for x in ft["date"]])
    for t in range(1, len(c)):
        if c[t-1] <= 0: continue
        j = didx_all.get(dates[t], -1)
        if j >= 0:
            tot[j] += (c[t]/c[t-1]-1.0)*PCT; cnt[j] += 1
mret = np.where(cnt > 0, tot/np.maximum(cnt, 1), 0.0)
mkt_cache = {"mret": mret, "mdidx": didx_all}

# ============ 逐信号体检 ============
for sname, fpath, sfield in SIGNALS:
    print("\n===== %s =====" % sname)
    STYLE_CACHE.clear()   # 同 code 在不同特征文件里阵列长度不同（不同构建批次），缓存必须按文件隔离
    d = pickle.load(open(os.path.join(HERE, fpath), "rb"))
    feats = d["feats"]
    pool = d.get("pool_codes") or set()

    # ---- 收集信号行 + 宇宙日截面 ----
    # day_univ[date] = list of (code, idx, r5, r10, r20, z-styles..., sig_flag)
    # 为内存效率：两遍——第一遍按日存（code, idx, r_h, styles, sig）
    rows = []                     # 信号行
    day_univ = defaultdict(list)  # date -> [rec]
    n_skip = 0
    for code, ft in feats.items():
        st = styles_of(code, ft, mkt_cache, BP1_RAW)
        if st is None:
            n_skip += 1
            continue
        sig = np.asarray(ft[sfield]).astype(bool)
        dates = st["dates"]; n = len(dates)
        for i in np.where(sig)[0]:
            ds = dates[i]
            if not (D0 <= ds <= D1): continue
            rows.append((code, int(i), ds))
        for i in range(n):
            ds = dates[i]
            if not (D0 <= ds <= D1): continue
            if np.isnan(st["r20"][i]) and np.isnan(st["r10"][i]) and np.isnan(st["r5"][i]):
                continue
            day_univ[ds].append((code, i))
    if n_skip:
        print("  [warn] 无对齐原始数据跳过 %d 只" % n_skip)
    print("  信号行 %d, 宇宙日 %d" % (len(rows), len(day_univ)))

    # ---- 模块A: 全截面 Pearson IC + RankIC（二元信号因子，5/10/20） ----
    icA = {"pearson": {h: [] for h in (5, 10, 20)}, "rank": {h: [] for h in (5, 10, 20)}}
    sig_by_day = defaultdict(set)
    for code, i, ds in rows:
        sig_by_day[ds].add((code, i))
    for ds, lst in day_univ.items():
        if len(lst) < 30: continue
        ssig = sig_by_day.get(ds, set())
        f = np.array([1.0 if (c, i) in ssig else 0.0 for c, i in lst])
        if f.sum() == 0 or f.std() == 0: continue
        for h in (5, 10, 20):
            rr = []
            keep = []
            for k, (c, i) in enumerate(lst):
                st = STYLE_CACHE[c]
                v = st["r%d" % h][i]
                if not np.isnan(v):
                    keep.append(k); rr.append(v)
            if len(rr) < 30: continue
            fh = f[keep]; rh = np.array(rr)
            if fh.std() == 0: continue
            try:
                pr, _ = pearsonr(fh, rh); icA["pearson"][h].append(pr)
                sr, _ = spearmanr(fh, rh); icA["rank"][h].append(sr)
            except Exception:
                pass

    # ---- 模块B: 风格暴露 + 风格调整 alpha ----
    exp_acc = defaultdict(list)   # style -> [daily mean z]
    abn_list = []; raw_list = []
    abn_train = []; abn_valid = []
    abn_by_day = {}               # ds -> 当日均值（用于按日聚合 t，规避截面相关膨胀）
    for ds, lst in day_univ.items():
        ssig = sig_by_day.get(ds, set())
        if not ssig: continue
        if len(lst) < 50: continue
        # 截面 z
        vals = {}
        for key in ("ln_amt", "beta", "mom60", "vol20", "rev20", "r20"):
            arr = np.array([STYLE_CACHE[c][key][i] for c, i in lst], dtype=float)
            vals[key] = arr
        zs = {}
        for key in ("ln_amt", "beta", "mom60", "vol20", "rev20"):
            zs[key] = zscore_cs(vals[key])
        sigmask = np.array([(c, i) in ssig for c, i in lst])
        for key in zs:
            z = zs[key][sigmask]
            z = z[~np.isnan(z)]
            if len(z): exp_acc[key].append(float(z.mean()))
        # 日截面 OLS: r20 ~ 1 + 5 z-styles（宇宙），得信号行异常
        y = vals["r20"]
        m = ~(np.isnan(y) | np.isnan(zs["ln_amt"]) | np.isnan(zs["beta"]) |
              np.isnan(zs["mom60"]) | np.isnan(zs["vol20"]) | np.isnan(zs["rev20"]))
        if m.sum() < 60: continue
        X = np.column_stack([np.ones(m.sum()), zs["ln_amt"][m], zs["beta"][m],
                             zs["mom60"][m], zs["vol20"][m], zs["rev20"][m]])
        ym = y[m]
        try:
            coef, *_ = np.linalg.lstsq(X, ym, rcond=None)
        except Exception:
            continue
        # 信号行（含缺失风格的丢弃）
        sm = m & sigmask
        if sm.sum() == 0: continue
        Xs = np.column_stack([np.ones(sm.sum()), zs["ln_amt"][sm], zs["beta"][sm],
                              zs["mom60"][sm], zs["vol20"][sm], zs["rev20"][sm]])
        pred = Xs @ coef
        abn = y[sm]*100.0 - pred*100.0
        abn_list.extend(abn.tolist())
        raw_list.extend((y[sm]*100.0).tolist())
        abn_by_day[ds] = float(np.mean(abn))
        if ds < TRAIN_END: abn_train.extend(abn.tolist())
        else: abn_valid.extend(abn.tolist())

    # ---- 模块C: 换手 ----
    pos_days = defaultdict(set)   # date -> set(code)
    sig_idx = defaultdict(set)    # code -> set(idx)
    rows_by_code = defaultdict(list)
    for code, i, ds in rows:
        rows_by_code[code].append(i)
    for code, idxs in rows_by_code.items():
        st = styles_of(code, feats[code], mkt_cache, BP1_RAW)
        if st is None: continue
        n = len(st["dates"])
        sig_idx[code].update(idxs)
        for i in idxs:
            e = i + 1; x = i + 20
            if e >= n or x >= n: continue
            for t in range(e, x+1):
                dd = st["dates"][t]
                if D0 <= dd <= D1:
                    pos_days[dd].add(code)
    pdates = sorted(pos_days)
    turns = []; overlaps = []
    for k in range(1, len(pdates)):
        p0 = pos_days[pdates[k-1]]; p1 = pos_days[pdates[k]]
        if not p1: continue
        new = len(p1 - p0); closed = len(p0 - p1)
        avg = 0.5*(len(p0)+len(p1))
        if avg > 0: turns.append((new+closed)/avg)
        overlaps.append(len(p0 & p1)/len(p1))
    # 信号粘性: P(sig[t+1] | sig[t])
    sticky_num = 0; sticky_den = 0
    for code, idxs in sig_idx.items():
        for i in idxs:
            sticky_den += 1
            if (i+1) in idxs: sticky_num += 1

    # ---- 模块D: NAV 自相关 ----
    drag = ROUND_TRIP/20.0
    dayret = defaultdict(list)
    codes_in_pos = set()
    for dd in pdates: codes_in_pos |= pos_days[dd]
    for code in codes_in_pos:
        st = STYLE_CACHE[code]
        c = st["c"]; dates = st["dates"]
        idx_of = {dates[i]: i for i in range(len(dates))}
        for dd in pdates:
            if code not in pos_days[dd]: continue
            i = idx_of.get(dd)
            if i is None or i < 1 or c[i-1] <= 0: continue
            dayret[dd].append((c[i]/c[i-1]-1.0)*PCT - drag)
    daily = {dd: float(np.mean(v)) for dd, v in dayret.items() if v}
    seq = [daily[dd] for dd in sorted(daily)]
    if len(seq) > 30:
        a = np.array(seq[:-1]); b = np.array(seq[1:])
        ac1 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    else:
        ac1 = None

    # ---- 汇总 ----
    def dstat(v):
        a = np.array(v, float); a = a[np.isfinite(a)]
        if len(a) == 0: return None
        return {"n": int(len(a)), "mean": float(a.mean()),
                "t": float(a.mean()/(a.std()/np.sqrt(len(a)))) if a.std() > 0 else None}

    # 按日聚合 t：每日信号行 abn 均值作为观测（消除同日截面相关对 t 的膨胀）
    dvals = np.array(list(abn_by_day.values()), float) if abn_by_day else np.array([])
    if len(dvals) > 2 and dvals.std() > 0:
        abn_t_daily = float(dvals.mean()/(dvals.std()/np.sqrt(len(dvals))))
    else:
        abn_t_daily = None
    R[sname] = {
        "n_sig": len(rows),
        "ic_full": {
            "pearson": {str(h): ic_block(icA["pearson"][h]) for h in (5, 10, 20)},
            "rank": {str(h): ic_block(icA["rank"][h]) for h in (5, 10, 20)},
        },
        "exposure": {k: dstat(v) for k, v in exp_acc.items()},
        "style_alpha": {
            "raw_mean": float(np.mean(raw_list)) if raw_list else None,
            "abn_mean": float(np.mean(abn_list)) if abn_list else None,
            "abn_t": float(np.mean(abn_list)/(np.std(abn_list)/np.sqrt(len(abn_list)))) if len(abn_list) > 2 and np.std(abn_list) > 0 else None,
            "abn_t_daily": abn_t_daily,
            "abn_days": int(len(dvals)),
            "abn_train": float(np.mean(abn_train)) if abn_train else None,
            "abn_valid": float(np.mean(abn_valid)) if abn_valid else None,
            "n": len(abn_list),
        },
        "turnover": {
            "mean_daily": float(np.mean(turns)) if turns else None,
            "median_daily": float(np.median(turns)) if turns else None,
            "overlap": float(np.mean(overlaps)) if overlaps else None,
            "stickiness": float(sticky_num/sticky_den) if sticky_den else None,
            "n_days": len(pdates),
        },
        "nav_ac1": ac1,
    }
    print("  IC20 pearson=%s rank=%s | 换手=%.3f overlap=%.3f stick=%.3f ac1=%s" % (
        R[sname]["ic_full"]["pearson"]["20"] and "%+.4f" % R[sname]["ic_full"]["pearson"]["20"]["mean"],
        R[sname]["ic_full"]["rank"]["20"] and "%+.4f" % R[sname]["ic_full"]["rank"]["20"]["mean"],
        R[sname]["turnover"]["mean_daily"] or 0, R[sname]["turnover"]["overlap"] or 0,
        R[sname]["turnover"]["stickiness"] or 0,
        ("%+.3f" % ac1) if ac1 is not None else "—"))
    print("  风格alpha: raw=%+.2f%% abn=%+.2f%% (t=%s) train=%+.2f valid=%+.2f" % (
        R[sname]["style_alpha"]["raw_mean"] or 0, R[sname]["style_alpha"]["abn_mean"] or 0,
        "%.2f" % R[sname]["style_alpha"]["abn_t"] if R[sname]["style_alpha"]["abn_t"] else "—",
        R[sname]["style_alpha"]["abn_train"] or 0, R[sname]["style_alpha"]["abn_valid"] or 0))
    del d, feats

json.dump(R, open(os.path.join(HERE, "healthcheck_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved -> healthcheck_results.json")
