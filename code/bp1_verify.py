# -*- coding: utf-8 -*-
"""
bp1因子重构评估 — Step2 校准验证
对象: bp12_rebacktest_trades.csv (2026-08-04, 9个季度采样日, 池内1242只, 47条bp1)
方法:
  1. 信号集比对: 同采样日 + 池内子集, 重放bp1_sig vs CSV 47条 (匹配/漏检/多检)
  2. 模拟复现: 按原引擎逻辑(T+1开盘入场, 15%止损, 峰值50%回撤止盈)复现pnl/ret3/ret10/ret20
  3. 统计比对: bp1_3d/10d 汇总指标
"""
import pickle, os, json, csv, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\Lenovo\WorkBuddy\2026-07-17-00-40-38"
POOL = r"C:\Users\Lenovo\WorkBuddy\2026-06-07-15-29-35\screener\stock_pool.json"
FEAT = os.path.join(HERE, "bp1_features_poolref.pkl")

d = pickle.load(open(os.path.join(HERE, "bp1_features_poolref84.pkl"), "rb"))
feats = d["feats"]
POOL_FILE = os.path.join(HERE, "_pool_20260804.json")
pool_codes = set(s["code"] for s in json.load(open(POOL_FILE, encoding="utf-8"))["stocks"])
print("features: %d stocks, pool(20260804) codes: %d" % (len(feats), len(pool_codes)))

# ---- 1. 重放信号(池内子集) ----
rows = list(csv.DictReader(open(os.path.join(SRC, "bp12_rebacktest_trades.csv"), encoding="utf-8")))
bp1_csv = [r for r in rows if r["label"].startswith("bp1")]
csv_set = set((r["symbol"], r["signal_date"]) for r in bp1_csv)
print("CSV bp1: %d 条, 采样日 %s" % (len(csv_set), sorted(set(d for _, d in csv_set))))

replay = set()
for code, ft in feats.items():
    if code not in pool_codes: continue
    dates = ft["date"]
    for i in np.where(ft["bp1_sig"])[0]:
        replay.add((code, str(dates[i])))
match = csv_set & replay
missed = csv_set - replay
sample_dates = set(d for _, d in csv_set)
extra = set(x for x in replay - csv_set if x[1] in sample_dates)   # 多检只统计采样日
print("\n[信号集比对] 匹配 %d / %d (%.1f%%) | 漏检 %d | 多检(采样日) %d" % (
    len(match), len(csv_set), 100*len(match)/len(csv_set), len(missed), len(extra)))
if missed:
    print("漏检样例:", sorted(missed)[:8])
if extra:
    print("多检样例:", sorted(extra)[:8])

# ---- 2. 模拟复现(原引擎逻辑) ----
def simulate(code, ft, sdate, stop_loss=0.15, trail_dd=0.50):
    dates = ft["date"]; o = ft["o"]; c = ft["c"]
    n = len(dates)
    si = None
    for i in range(n):
        if str(dates[i]) == sdate: si = i; break
    if si is None or si+1 >= n: return None
    entry = float(o[si+1])
    if entry <= 0: return None
    exit_price = entry; exit_reason = "holding"
    peak = entry; ret3 = ret10 = ret20 = None
    for j in range(si+2, n):
        cc = float(c[j])
        bars = j-(si+1)
        if bars == 2: ret3 = round((cc/entry-1)*100, 2)
        if bars == 9: ret10 = round((cc/entry-1)*100, 2)
        if bars == 19: ret20 = round((cc/entry-1)*100, 2)
        if cc < entry*(1-stop_loss):
            exit_price = cc; exit_reason = "stop"; break
        if cc > peak: peak = cc
        if cc < peak*(1-trail_dd):
            exit_price = cc; exit_reason = "trailing"; break
    return {"pnl": round((exit_price/entry-1)*100, 2), "ret3": ret3, "ret10": ret10, "ret20": ret20,
            "exit_reason": exit_reason, "entry": round(entry, 3)}

print("\n[模拟复现] 匹配信号按原口径(15%止损+50%回撤止盈):")
ok, bad = 0, 0
diffs = []
for code, sdate in sorted(match):
    csv_row = next(r for r in bp1_csv if r["symbol"] == code and r["signal_date"] == sdate)
    sim = simulate(code, feats[code], sdate)
    if sim is None:
        bad += 1; continue
    csv_pnl = float(csv_row["pnl_pct"]) if csv_row["pnl_pct"] not in ("", "N/A") else None
    if csv_pnl is not None:
        diffs.append(abs(sim["pnl"]-csv_pnl))
        if abs(sim["pnl"]-csv_pnl) < 0.05: ok += 1
        else: bad += 1
if diffs:
    diffs = np.array(diffs)
    print("  可比对 %d 条: |pnl差|<0.05pct 的 %d 条, 中位差 %.3f pct, 均值差 %.3f pct, 最大差 %.2f pct" % (
        len(diffs), ok, np.median(diffs), diffs.mean(), diffs.max()))

# ---- 3. 汇总统计比对 ----
import statistics as st
sim_pnls = []
for code, sdate in sorted(match):
    sim = simulate(code, feats[code], sdate)
    if sim and sim["pnl"] is not None: sim_pnls.append(sim["pnl"])
csv_pnls = [float(r["pnl_pct"]) for r in bp1_csv if r["pnl_pct"] not in ("", "N/A")]
print("\n[汇总] 我方重放(匹配集, n=%d): mean=%+.2f%% median=%+.2f%% win=%.1f%%" % (
    len(sim_pnls), st.mean(sim_pnls), st.median(sim_pnls), 100*sum(1 for x in sim_pnls if x > 0)/len(sim_pnls)))
print("[汇总] CSV原文 (n=%d): mean=%+.2f%% median=%+.2f%% win=%.1f%%" % (
    len(csv_pnls), st.mean(csv_pnls), st.median(csv_pnls), 100*sum(1 for x in csv_pnls if x > 0)/len(csv_pnls)))
print("[汇总] CSV meta: bp1_3d_avg=-2.061 bp1_3d_win=34.04 bp1_10d_avg=0.896 bp1_10d_win=38.3")

# ---- 4. 漏检归因 ----
if missed:
    print("\n[漏检归因] 逐条诊断:")
    for code, sdate in sorted(missed)[:20]:
        ft = feats.get(code)
        if ft is None:
            print("  %s %s: 不在我的universe(名单外/指数/ K线不足)" % (code, sdate)); continue
        dates = ft["date"]; si = None
        for i in range(len(dates)):
            if str(dates[i]) == sdate: si = i; break
        if si is None:
            print("  %s %s: 该日无K线" % (code, sdate)); continue
        tech = bool(ft["bp1_tech"][si]); l0 = bool(ft["l0_pass"][si])
        print("  %s %s: tech=%s l0=%s n_raw=%s rps=%.1f d1=%.0f d2=%.0f d3=%.0f d4=%.0f stage=%.2f" % (
            code, sdate, tech, l0, ft["n_raw"][si], ft["rps"][si] if not np.isnan(ft["rps"][si]) else -1,
            ft["d1"][si], ft["d2"][si], ft["d3"][si], ft["d4"][si], ft["stage_mult"][si]))

json.dump({"match": len(match), "csv": len(csv_set), "missed": len(missed), "extra": len(extra),
           "missed_list": sorted(missed), "extra_list": sorted(extra)},
          open(os.path.join(HERE, "bp1_verify_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved -> bp1_verify_result.json")
