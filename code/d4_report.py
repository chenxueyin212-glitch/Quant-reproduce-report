# -*- coding: utf-8 -*-
"""
d4 伏击因子（波动率压缩埋伏）深度评估报告 — S级产品化独立报告
信号: d4>=15 & platform_range<25% & ~bp1_tech（未突破）
口径: T日收盘后可知 → T+1开盘入场 → T+20收盘出场(t1_20)，训练<20250908，费率0.30%/往返
数据: d4_full_results.json + d4_nav_equity.json（d4_full_analysis.py 产出）
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "d4_full_results.json"), encoding="utf-8"))
EQ = json.load(open(os.path.join(HERE, "d4_nav_equity.json"), encoding="utf-8"))

def f2(x): return ("%+.2f" % x) if x is not None else "—"
def f1(x): return ("%+.1f" % x) if x is not None else "—"
def cls(x): return "up" if (x or 0) > 0 else "down"

def bar_chart(groups, width=940, height=None, pad=(56, 14, 30, 60), fmt="%+.2f"):
    ns = max(len(g[1]) for g in groups)
    bh = 22; gap = 7; gh = ns*bh+(ns-1)*2+gap
    height = height or (pad[2]+len(groups)*gh+pad[3])
    pl, pr, pt, pb = pad
    iw, ih = width-pl-pr, height-pt-pb
    vals = [v for _, ss in groups for _, _, v in ss]
    lo, hi = min(vals+[0]), max(vals+[0])
    rng = (hi-lo) or 1
    lo -= rng*0.12; hi += rng*0.18
    def Y(v): return pt+ih*(hi-v)/(hi-lo)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="width:100%%;height:auto">' % (width, height)]
    for g in range(5):
        v = lo+(hi-lo)*g/4; y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb"/>' % (pl, y, pl+iw, y))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%s</text>' % (pl-6, y+4, fmt % v))
    y0 = Y(0)
    out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#9ca3af" stroke-width="1.2"/>' % (pl, y0, pl+iw, y0))
    bw = 96
    for gi, (label, ss) in enumerate(groups):
        gx = pl+iw*(gi+0.5)/len(groups)
        total_w = len(ss)*bw+(len(ss)-1)*2
        for si, (sn, color, v) in enumerate(ss):
            bx = gx-total_w/2+si*(bw+2)
            yv = Y(v)
            out.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" fill="%s" rx="2"/>' % (bx, min(yv, y0), bw, abs(y0-yv), color))
            ty = yv-5 if v >= 0 else yv+13
            out.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" text-anchor="middle" font-weight="600">%s</text>' % (bx+bw/2, ty, color, fmt % v))
        out.append('<text x="%.1f" y="%d" font-size="11.5" fill="#374151" text-anchor="middle">%s</text>' % (gx, height-8, label))
    lx = pl; seen = []
    for _, ss in groups:
        for sn, color, _ in ss:
            if (sn, color) not in seen: seen.append((sn, color))
    for sn, color in seen:
        out.append('<rect x="%d" y="6" width="10" height="10" fill="%s" rx="2"/>' % (lx, color))
        out.append('<text x="%d" y="15" font-size="11" fill="#374151">%s</text>' % (lx+14, sn))
        lx += 20+len(sn)*12
    out.append("</svg>")
    return "".join(out)

def line_chart(series, width=940, height=300, pad=(56, 18, 34, 70), ymin=None, ymax=None, vline=None, vlabel=None):
    pl, pr, pt, pb = pad
    iw, ih = width-pl-pr, height-pt-pb
    ys = [v for _, _, d in series for v in d.values()]
    lo = ymin if ymin is not None else min(ys)*0.97
    hi = ymax if ymax is not None else max(ys)*1.03
    xs_all = sorted({k for _, _, d in series for k in d})
    def X(i): return pl + iw*i/(len(xs_all)-1)
    def Y(v): return pt + ih*(hi-v)/(hi-lo)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="width:100%%;height:auto">' % (width, height)]
    for g in range(5):
        v = lo+(hi-lo)*g/4; y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb" stroke-width="1"/>' % (pl, y, pl+iw, y))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#6b7280" text-anchor="end">%.0f</text>' % (pl-6, y+4, v))
    for i in range(0, len(xs_all), 3):
        out.append('<text x="%.1f" y="%d" font-size="10.5" fill="#6b7280" text-anchor="middle">%s</text>' % (X(i), height-10, xs_all[i]))
    if vline and vline in xs_all:
        i = xs_all.index(vline)
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#d97706" stroke-width="1.4" stroke-dasharray="5,4"/>' % (X(i), pt, X(i), pt+ih))
        out.append('<text x="%.1f" y="%d" font-size="11" fill="#d97706" text-anchor="middle">%s</text>' % (X(i), pt-6, vlabel or ""))
    for name, color, d in series:
        pts = " ".join("%.1f,%.1f" % (X(xs_all.index(k)), Y(v)) for k, v in sorted(d.items()) if k in xs_all)
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.2"/>' % (pts, color))
        lastk = sorted(d.keys())[-1]
        out.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (X(xs_all.index(lastk)), Y(d[lastk]), color))
        out.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="%s" font-weight="600">%s</text>' % (X(xs_all.index(lastk))+5, Y(d[lastk])+4, color, name))
    out.append("</svg>")
    return "".join(out)

# ================= 数据装配 =================
g = R["groups"]; nv = R["naive"]; dc = R["deciles"]
nav = R["nav"]; port = R["port"]; pool_port = R["pool_port"]; bench_port = R["bench_port"]
ofat = R["ofat"]; regime = R["regime"]; cp = R["counterparty"]; qt = R["quality"]; path = R["path"]
ic = R["ic"]; mae = R["mae_mismatch"]; monthly = R["monthly"]

def row(label, d, bold=False):
    if not d: return ""
    st = ' style="font-weight:700"' if bold else ""
    return ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td>"
            "<td>%s%%</td><td>%s%%</td></tr>") % (
        st, label, "{:,}".format(d["n"]), cls(d["mean"]), f2(d["mean"]), cls(d["median"]), f2(d["median"]),
        f1(d["win"]), f1(d["p10"]), f1(d["p90"]))

rows_groups = "".join([
    row("全体", g["全体"], True),
    row("训练期（<20250908）", g["训练期"]),
    row("验证期（≥20250908）", g["验证期"]),
    row("池内（848 只）", g["池内"]),
    row("池外", g["池外"]),
    row("池内-训练", g["池内-训练期"]),
    row("池内-验证", g["池内-验证期"]),
    row("池外-训练", g["池外-训练期"]),
    row("池外-验证", g["池外-验证期"]),
    row("run = 1（fresh 初现）", g["run=1(fresh)"]),
    row("run 2-5", g["run 2-5"]),
    row("run > 5（sustained 持续）", g["run >5(sustained)"]),
])

rows_naive = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
    for k, v in nv.items() if v)

rows_decile = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]),
        ("%.1f%%" % v["pnl_share"]) if v.get("pnl_share") is not None else "—")
    for k, v in dc.items())

# IC 表
icu = ic["universe_d4"]; icg = ic["signal_group_d4"]; icp = ic["signal_group_plat"]
ic_rows = ""
for lbl, v in [("全宇宙日截面：d4 (0–15) vs 20日收益", icu),
               ("信号组内：d4 vs 20日收益", icg),
               ("信号组内：platform_range vs 20日收益", icp)]:
    if v.get("mean") is None:
        ic_rows += ("<tr><td>%s</td><td colspan='5'><span class='tag tag-gray'>%s</span></td></tr>") % (lbl, v.get("note", "—"))
    else:
        tag = "正" if v["mean"] > 0 else "负"
        sig = "显著" if abs(v.get("t") or 0) > 2 else "不显著"
        ic_rows += ("<tr><td>%s</td><td class='%s'>%+.4f</td><td>%+.3f</td><td>%.1f%%</td><td class='%s'>%+.2f</td>"
                    "<td><span class='tag %s'>%s且%s</span></td></tr>") % (
            lbl, cls(v["mean"]), v["mean"], v["icir"], v["pos_ratio"], cls(v["t"]), v["t"],
            "tag-green" if v["mean"] > 0 else "tag-red", tag, sig)

# cohort 组合表
port_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%"]:
    v = port[k]; a = v["all"]; tr = v["train"]; va = v["valid"]
    if not a: continue
    port_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]), f1(a["win_m"]),
        cls(tr["total"]), f1(tr["total"]), cls(va["total"]), f1(va["total"]))
bp = bench_port["等权全A"]
port_rows += ("<tr style='background:#fdecea'><td><strong>基准·等权全A（月度cohort）</strong></td><td class='up'><strong>%s%%</strong></td>"
              "<td><strong>%s</strong></td><td><strong>%s%%</strong></td><td><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td></tr>") % (
    f1(bp["all"]["total"]), f2(bp["all"]["sharpe"]), f1(bp["all"]["maxdd"]), f1(bp["all"]["win_m"]),
    f1(bp["train"]["total"]), f1(bp["valid"]["total"]))

pool_rows = ""
for k in ["不止损", "止损8%"]:
    v = pool_port[k]
    for seg, lbl in [("all", "全部"), ("train", "训练期"), ("valid", "验证期")]:
        a = v[seg]
        if not a: continue
        pool_rows += ("<tr><td>%s / %s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td><td>%s%%</td></tr>") % (
            k, lbl, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]), f1(a["win_m"]))

# NAV 表
nav_rows = ""
order = ["d4_全A", "基准_等权全A", "d4_池内", "基准_等权池内", "d4_池外", "基准_等权池外"]
for k in order:
    m = nav[k]["metrics"]
    style = " style='background:#fdecea'" if k.startswith("基准") else ""
    nm = ("<strong>%s</strong>" % k) if k.startswith("基准") else k
    nav_rows += ("<tr%s><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                 "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        style, nm,
        cls(m["全部"]["total"]), f1(m["全部"]["total"]), f2(m["全部"]["sharpe"]), f1(m["全部"]["maxdd"]),
        cls(m["训练期"]["total"]), f1(m["训练期"]["total"]),
        cls(m["验证期"]["total"]), f1(m["验证期"]["total"]))

# NAV 月度曲线（净值→月初指数化）
def to_monthly_idx(eq):
    out = {}
    for dd in sorted(eq.keys()):
        if not ("20231101" <= dd <= "20260821"): continue
        ym = dd[:6]
        out[ym] = eq[dd]
    base = out[sorted(out.keys())[0]]
    return {k: v/base*100 for k, v in out.items()}

nav_chart = line_chart([
    ("d4_池内", "#e74c3c", to_monthly_idx(EQ["d4_池内"])),
    ("基准·池内", "#2563eb", to_monthly_idx(EQ["基准_等权池内"])),
    ("d4_全A", "#d97706", to_monthly_idx(EQ["d4_全A"])),
    ("基准·全A", "#6b7280", to_monthly_idx(EQ["基准_等权全A"])),
], vline="202509", vlabel="训练|验证分界")

# OFAT 表
ofat_rows = ""
for k, v in ofat.items():
    if not v: continue
    bold = k.startswith("基准")
    st = ' style="font-weight:700"' if bold else ""
    ofat_rows += ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>") % (
        st, k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))

regime_rows = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))
    for k, v in regime.items() if v)

def layer_rows(block):
    out = ""
    for dim, grp in block.items():
        for k, v in grp.items():
            if not v: continue
            out += ("<tr><td>%s</td><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>") % (
                dim, k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
    return out

cp_rows = layer_rows(cp)
q_rows = layer_rows(qt)

path_rows = "".join(
    "<tr><td>%s</td><td>%+.2f%%</td><td>%+.2f%%</td></tr>" % (lbl, v["median"], v["mean"])
    for lbl, v in [("MAE 20日最大不利偏移", path["mae20"]), ("MFE 20日最大有利偏移", path["mfe20"]),
                   ("前 5 日最低", path["first5_min"]), ("后 15 日最高", path["last15_max"]),
                   ("低于成本天数（/20）", path["below_days"])])

mae_rows = "".join(
    "<tr><td>MAE ≤ −%s%%</td><td>%.1f%%</td></tr>" % (k, v) for k, v in mae["frac_below"].items())

# 月度柱状图（近 18 月）
mon_chart = bar_chart([
    (m[:4]+"/"+m[4:], [("伏击月均", "#e74c3c" if v["mean"] >= 0 else "#27ae60", v["mean"])])
    for m, v in list(monthly.items())[-18:]
])

# 分层对照柱状图
layer_chart = bar_chart([
    ("训练期", [("伏击", "#e74c3c", g["训练期"]["mean"]), ("宇宙", "#6b7280", nv["全宇宙-训练"]["mean"])]),
    ("验证期", [("伏击", "#e74c3c", g["验证期"]["mean"]), ("宇宙", "#6b7280", nv["全宇宙-验证"]["mean"])]),
    ("池内", [("伏击", "#e74c3c", g["池内"]["mean"]), ("池外", "#6b7280", g["池外"]["mean"])]),
    ("fresh vs 持续", [("run=1", "#d97706", g["run=1(fresh)"]["mean"]), ("run>5", "#e74c3c", g["run >5(sustained)"]["mean"])]),
])

ofat_chart = bar_chart([
    ("d4>=12", [("放宽", "#9ca3af", ofat["d4>=12"]["mean"]), ("基准>=15", "#e74c3c", ofat["基准(d4>=15,plat<25)"]["mean"])]),
    ("plat 门限", [("<20%", "#2563eb", ofat["plat<20%"]["mean"]), ("<25%基准", "#e74c3c", ofat["基准(d4>=15,plat<25)"]["mean"]), ("<30%", "#9ca3af", ofat["plat<30%"]["mean"])]),
    ("持有期", [("10日", "#9ca3af", ofat["持有10日"]["mean"]), ("20日基准", "#e74c3c", ofat["基准(d4>=15,plat<25)"]["mean"]), ("30日", "#2563eb", ofat["持有30日"]["mean"])]),
    ("含突破对照", [("含tech", "#9ca3af", ofat["含突破(不排除tech)"]["mean"]), ("排除tech", "#e74c3c", ofat["基准(d4>=15,plat<25)"]["mean"])]),
])

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>d4 伏击因子（波动率压缩埋伏）深度评估报告 · S级产品化（2026-10-20）</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--text:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--red:#e74c3c;--green:#27ae60;--blue:#2563eb;--amber:#d97706;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);line-height:1.75;padding:24px}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:25px;margin-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
h2{font-size:20px;margin:36px 0 14px;padding-left:10px;border-left:4px solid var(--blue)}
h3{font-size:16px;margin:20px 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
th{background:#f0f3f7;text-align:left;padding:7px 9px;font-weight:600;border-bottom:2px solid var(--line);white-space:nowrap}
td{padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
.up{color:var(--red);font-weight:600}
.down{color:var(--green);font-weight:600}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}
.tag-red{background:#fdecea;color:var(--red)}
.tag-green{background:#e8f8f5;color:var(--green)}
.tag-gray{background:#f0f3f7;color:var(--muted)}
.tag-amber{background:#fef3e2;color:var(--amber)}
.note{background:#fef9e7;border-left:4px solid var(--amber);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.warn{background:#fdecea;border-left:4px solid var(--red);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.key{background:#eef4ff;border-left:4px solid var(--blue);padding:10px 14px;border-radius:6px;font-size:13px;margin:12px 0}
.muted{color:var(--muted);font-size:12px}
code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:12.5px}
.formula{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:12px 16px;font-size:13px;margin:8px 0}
@page{margin:11mm 9mm;size:A4}
@media print{
body{padding:0;background:#fff;font-size:12px}
.wrap{max-width:100%}
.card{break-inside:avoid;page-break-inside:avoid;border:1px solid #d1d5db}
table{break-inside:avoid;page-break-inside:avoid;font-size:11px}
th,td{padding:5px 7px}
h2{break-after:avoid;page-break-after:avoid;margin:22px 0 10px}
h3{break-after:avoid;page-break-after:avoid}
svg{break-inside:avoid;page-break-inside:avoid}
.note,.warn,.key{break-inside:avoid;page-break-inside:avoid}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
</style>
</head>
<body>
<div class="wrap">

<h1>d4 伏击因子（波动率压缩埋伏）深度评估报告 <span class="tag tag-amber">S级产品化 · 独立章</span></h1>
<div class="sub">生成日期：2026-10-20 ｜ 方法：A股量化研报复现专家团 Workflow B ｜ 数据：kline_cache_all.db（前复权日线，2023-11-01 ~ 2026-08-21，全A 4999 只）｜ 验收：role-07 ｜ 来源：bp1 章 A 级修复 → 本章产品化完整五项检验</div>

<div class="warn">
<strong>验收结论先行（冷酷版）：产品化不通过，降级为「池内择时插件」。</strong>信号层优势真实且稳健——伏击 <span class="up">+2.65%</span>/20日 vs 全宇宙等权 <span class="up">+1.41%</span>（+1.24 pct 边际），全宇宙 d4 IC <span class="up">+0.039</span>（t=+8.8），验证期 +1.09% vs 宇宙 +0.54%（边际不消失）。<strong>但组合层产品化失败</strong>：日度等权再平衡 NAV（含成本）d4_全A 仅 <span class="down">+6.2%</span> vs 等权全A基准 <span class="up">+31.6%</span>——跑输 25.4 pct，验证期 <span class="down">−8.9%</span> vs 基准 +0.1%。机制：信号密度太高（日均活跃仓位 4,118 个、峰值 19,585），组合本质是「稀释的市场暴露」，每笔 +1.24 pct 的边际被持仓重叠与再平衡成本摊薄。<strong>唯一保留项：池内版本</strong>——d4_池内 NAV <span class="up">+51.3%</span>（验证期 <span class="up">+26.5%</span>，夏普 0.94），池内-验证期信号层 <span class="up">+3.37%</span>（n=3,883）<strong>样本外不衰减</strong>。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>"波动率压缩埋伏"的微观假设完全成立——<strong>同一套 d4 信息，埋伏 +2.65% vs 突破后追入 −0.29%（差 2.94 pct/20日）</strong>，平台越窄越好（plat&lt;15% 组 +2.76%）、压缩越持久越好（run&gt;5 组 +3.48% vs fresh +1.91%）、60 日已翻倍的不碰（−3.86%）。<strong>方向对 ≠ 产品成立</strong>：全A 等权组合跑输基准 25 pct，因为 155,275 条信号同时在场数千个仓位，摊薄成了 beta 的影子。<strong>能带走的产品形态：选股池内的伏击择时</strong>（池内验证期 +3.37%/笔、NAV +26.5%），不是全A伏击策略。
</div>

<!-- ============ 一、定义 ============ -->
<h2>一、因子定义与来源（bp1 报告 A 级修复的产品化）</h2>
<div class="card">
<p>背景：bp1（B1 倒车买点）报告微观检验发现「突破越猛越差、量放越猛越差」，d4 波动率压缩是 N 类评分中唯一正判别维度（奖励突破前的静默）。A 级修复假设：<strong>不追突破，在波动率压缩态埋伏</strong>。本章按完整五项检验（信号层 / 组合层 / 稳健性 / 微观 / role-07 验收）做产品化评估。</p>
<div class="formula">
<strong>伏击信号（d4_full_analysis.py · mask_d4）：</strong><br>
· <code>d4 ≥ 15</code>　—— d4 为 0–15 整数波动率压缩评分（bp1 N 类评分体系），<strong>15 为封顶值</strong><br>
· <code>platform_range &lt; 25%</code>　—— 平台振幅仍窄（未走扩）<br>
· <code>~bp1_tech</code>　—— 未触发 bp1 技术五条件（未突破）<br><br>
<strong>口径：</strong>T 日收盘后可知 → T+1 开盘入场 → T+20 收盘出场；训练期 &lt; 20250908；费率 0.30%/往返（NAV 口径）。
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>d4 门槛</td><td>≥ 15（封顶 15）</td><td>平台振幅</td><td>&lt; 25%</td></tr>
<tr><td>突破过滤</td><td>排除 bp1_tech（未突破态）</td><td>持有期</td><td>20 个交易日（OFAT 含 10/30 日）</td></tr>
<tr><td>信号总量</td><td>155,275 条（原始 175,610，去右截断 20,335 = 11.6%）</td><td>宇宙</td><td>全A 4,999 只（池内 848）</td></tr>
<tr><td>信号日</td><td>T 日收盘后可知</td><td>入场 / 出场</td><td>T+1 开盘 / T+20 收盘</td></tr>
</table>
<div class="note"><strong>口径事实（必须先说清）：</strong>① <strong>d4 封顶 15</strong>——信号组内 d4 恒等于 15，组内 IC 退化不可定义（见 3.3）；② <strong>~bp1_tech 过滤几乎不绑定</strong>——"含突破"对照 n=155,371 vs 基准 155,275，仅差 96 条：d4≥15 且 plat&lt;25 的压缩态与 bp1 突破态在定义上几乎互斥，伏击 vs 追入的差异来自<strong>两组不同信号</strong>（压缩态 vs 突破态），不是这个过滤器本身。</div>
</div>

<!-- ============ 二、口径 ============ -->
<h2>二、九项口径声明（与 B1/B2/bp1/补票章一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T 日收盘后可知 → T+1 开盘入场 → T+20 收盘出场（t1_20 = c[T+20]/o[T+1] − 1）</td></tr>
<tr><td>2</td><td>分组数</td><td>12 组分层 + 十分位 + run 三档 + 池内外 × 训练验证交叉</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头；对照基准为同宇宙等权持有</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除指数与名单外；未做 ST/涨跌停/停牌过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾</td></tr>
<tr><td>6</td><td>中性化</td><td>不做行业/市值中性化（D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（4,999 只，主基准）/ 等权池内（848 只，含后视偏差）</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（月度夏普 = 均值/标准差×√12；NAV 夏普按 244 日年化）</td></tr>
<tr><td>9</td><td>费率</td><td>信号层不计成本；NAV 口径含 0.30%/往返（入场/出场各扣 0.15%）</td></tr>
</table>
</div>

<!-- ============ 三、信号层 ============ -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分层总表（20 日持有收益，%）</h3>
<table>
<tr><th>分层</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>p10</th><th>p90</th></tr>
__ROWS_GROUPS__
</table>
__LAYER_CHART__
<div class="key"><strong>三个核心事实：</strong>① <strong>边际真实</strong>——全体 +2.65% vs 同期宇宙 +1.41%，+1.24 pct/20日；② <strong>验证期边际不消失但衰减</strong>——+1.09% vs 宇宙 +0.54%（+0.55 pct），中位数 −0.54% 说明靠右尾撑着；③ <strong>持续态显著强于初现态</strong>——run&gt;5 +3.48%/60.3% vs run=1 +1.91%/50.7%：压缩越持久、积蓄的弹性越大，与"埋伏"机制自洽。</div>

<h3>3.2 朴素对照</h3>
<table>
<tr><th>对照</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__ROWS_NAIVE__
</table>

<h3>3.3 日度截面 IC</h3>
<table>
<tr><th>检验</th><th>IC 均值</th><th>ICIR</th><th>正比例</th><th>t 值</th><th>判定</th></tr>
__IC_ROWS__
</table>
<div class="note"><strong>IC 口径说明（D30）：</strong>信号组内 d4 退化（d4 封顶 15 → 截面 std=0）无法计算组内 IC，改以<strong>全宇宙日截面 IC</strong> 为主检验：+0.0386（t=+8.79，ICIR 0.337，正比例 68.3%）——d4 作为全 A 排序因子有效但强度中等（对照：bp1 d4 秩相关 +0.141 是组内口径，不可直接比）。组内 plat IC −0.051（t=−4.43）：<strong>平台越窄未来越好</strong>，与信号门槛方向一致。</div>

<h3>3.4 十分位（按 20 日收益排序）</h3>
<table>
<tr><th>十分位</th><th>n</th><th>均值</th><th>盈亏占比</th></tr>
__ROWS_DECILE__
</table>
<div class="warn"><strong>彩票型分布：</strong>D10（+30.74%）独占 116% 的总盈亏，D1−D4 四档合计 −117.6%——<strong>去掉 top 10% 赢家后全组转负</strong>（与 B1/bp1/补票同构）。中位数 +1.14% 远低于均值 +2.65%，典型右偏。可操作含义：出场机制（移动止盈保右尾）比入场筛选更重要。</div>
</div>

<!-- ============ 四、组合层 ============ -->
<h2>四、组合层绩效（月度 cohort + 日度 NAV 双口径）</h2>
<div class="card">
<h3>4.1 月度 cohort（每月信号等权持 20 日，不含成本）</h3>
<table>
<tr><th>止损档</th><th>累计</th><th>夏普</th><th>最大回撤</th><th>月胜率</th><th>训练期</th><th>验证期</th></tr>
__PORT_ROWS__
</table>
<div class="note">止损 8% 档最优（+14.8%，夏普 0.32）——MAE 中位 −4.86%、8% 档仅触发 30.8%，止损切的是左尾（形态健康，与补票同构、与 bp1 相反）。但<strong>全A cohort 各档全部跑输基准 cohort（+56.8%）</strong>，验证期（2025-09 后）全A 口径全部转负。</div>

<h3>4.2 月度 cohort：池内限定</h3>
<table>
<tr><th>止损档 / 区间</th><th>累计</th><th>夏普</th><th>最大回撤</th><th>月胜率</th></tr>
__POOL_ROWS__
</table>

<h3>4.3 日度等权再平衡 NAV（bp1 同口径，含成本 0.30%/往返）</h3>
<table>
<tr><th>组合</th><th>全部</th><th>夏普</th><th>最大回撤</th><th>训练期</th><th>验证期</th></tr>
__NAV_ROWS__
</table>
__NAV_CHART__
<div class="warn"><strong>产品化否决点就在这里：</strong>信号层每笔 +1.24 pct 的边际，到 NAV 层变成 <strong>+6.2% vs +31.6%</strong>（跑输 25.4 pct）。归因（4.4）：信号密度。而<strong>池内 NAV +51.3%（验证期 +26.5%、夏普 0.94）</strong>保留了可用性——池内池外差 51.7 pct，alpha 主要来自选股池 × 伏击的交互，不是伏击本身。</div>

<h3>4.4 组合层与信号层的背离归因</h3>
<table>
<tr><th>诊断项</th><th>数值</th><th>含义</th></tr>
<tr><td>日均活跃仓位</td><td>4,118 个（中位 2,182 / p90 12,577 / 峰值 19,585）</td><td>组合同时持有数千只，≈ 稀释的全A等权</td></tr>
<tr><td>仓位加权日均收益</td><td>+0.153%/日（约合 +3.0%/20日）</td><td>每笔持仓的平均日贡献是正的</td></tr>
<tr><td>等权日均收益（组合口径）</td><td>+0.035%/日</td><td>再平衡 + 重叠后落到组合上的只剩零头</td></tr>
<tr><td>稀疏日（&lt;500 仓）</td><td>155 天，日均 +0.077%</td><td>低密度日反而更干净</td></tr>
<tr><td>密集日（≥2,000 仓）</td><td>369 天，日均 +0.120%</td><td>高密度日贡献摊薄后的收益</td></tr>
<tr><td>月度信号量分布</td><td>263 条（2024-04）~ 16,948 条（2025-06）</td><td>cohort 月度等权给每月同权重，与 NAV 口径方向可背离</td></tr>
</table>
<div class="key"><strong>结论：</strong>伏击信号是"选股节拍器"不是"组合策略"——当它一天点名 4,000+ 只股票时，任何等权组合都退化为市场 beta 的高换手变体（成本还更高）。<strong>产品化必须先把密度打下来</strong>（见第九章：池内限定 / 限量 top-N / run≥6 过滤）。</div>
</div>

<!-- ============ 五、稳健性 ============ -->
<h2>五、稳健性检验（OFAT + regime）</h2>
<div class="card">
<h3>5.1 单因素敏感性（OFAT）</h3>
<table>
<tr><th>变体</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__OFAT_ROWS__
</table>
__OFAT_CHART__
<div class="note"><strong>参数面平缓（好事）：</strong>d4 门槛从 15 放宽到 12（n×2.7）均值只从 +2.65% 降到 +2.06%；plat 门限 20/25/30% 几乎不动。<strong>持有期越长越好</strong>（30 日 +4.74% vs 20 日 +2.65% vs 10 日 +1.47%）——弹性释放需要时间，与"压缩积蓄弹性"机制自洽，也提示出场可以更慢。含/排除突破几乎无差（定义互斥，见第一章注）。</div>

<h3>5.2 市场状态条件</h3>
<table>
<tr><th>条件</th><th>n</th><th>均值</th><th>胜率</th></tr>
__REGIME_ROWS__
</table>
<div class="warn"><strong>硬门禁候选：vol20 ≥ 3% 组 <span class="down">−0.55%</span>（胜率 38.4%）</strong>——高波动环境直接破坏"压缩积蓄弹性"的前提（没有压缩就没有弹性）。pre20 正负无关（+2.66% vs +2.64%）——伏击不挑短期动量方向，只挑压缩质量。</div>
</div>

<!-- ============ 六、微观 ============ -->
<h2>六、微观经济假设检验（伏击到底在买什么）</h2>
<div class="card">
<h3>6.1 对手方分层</h3>
<table>
<tr><th>维度</th><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__CP_ROWS__
</table>
<div class="key"><strong>接盘边界清晰：</strong>pre60 ≥ 100%（60 日翻倍）组 <span class="down">−3.86%</span>（胜率 24.1%）——压缩之后等着你的是出货不是弹性。<strong>硬门禁：pre60 &lt; 50%。</strong>pre60 &lt; 0% 组 +3.14% 最好——阴跌后的压缩最扎实。dd20 分层平缓（−10~−5% 组最好 +2.94%），深回撤（&lt;−15%）+2.31% 仍为正——伏击对回撤深度不敏感（与补票的 dd20&gt;−15% 门禁不同，伏击的平台条件已经替它挡掉了破位股）。</div>

<h3>6.2 质量分层（压缩质量）</h3>
<table>
<tr><th>维度</th><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__Q_ROWS__
</table>
<div class="key"><strong>H1（压缩积蓄弹性）全面验证：</strong>plat 越窄越好单调（&lt;15%：+2.76% → 15-20%：+2.35% → ≥20%：+1.90%）；run 越久越好单调（1：+1.91% → 2-5：+2.40% → &gt;5：<span class="up">+3.48%</span>）。<strong>质量阶梯明确：持续 × 窄平台 × 阴跌后</strong>是最优象限，与 bp1 报告"奖励突破前的静默"完全一致。</div>

<h3>6.3 持仓路径</h3>
<table>
<tr><th>指标</th><th>中位</th><th>均值</th></tr>
__PATH_ROWS__
</table>
<table>
<tr><th>MAE 档位</th><th>触发比例</th></tr>
__MAE_ROWS__
</table>
<div class="note">MAE 中位 −4.86% / MFE 中位 +7.18% / 前 5 日最低 −1.46% / 后 15 日最高 +4.98% / 低于成本天数中位 8/20。<strong>持仓体验五章最好</strong>：入场后前 5 日回撤浅（埋伏不追高），后 15 日弹性逐步释放（last15_max 中位 +4.98%）。8% 止损触发率 30.8%（补票 39.9%、bp1 54.7%）——左尾更薄。</div>

<h3>6.4 月度分布（近 18 月）</h3>
__MON_CHART__
<div class="muted">2025-05 ~ 2025-08 连续正月（+2.99/+5.98/+8.02/+2.94）是伏击的黄金段；2026-05（−11.14%）、2026-02（−6.72%）提示验证期后段环境转差。</div>
</div>

<!-- ============ 七、偏离登记 ============ -->
<h2>七、偏离登记（继承 D1–D28，本章新增）</h2>
<div class="card">
<table>
<tr><th>#</th><th>偏离项</th><th>说明</th><th>影响</th><th>等级</th></tr>
<tr><td>D1/D2/D12</td><td>无中性化 / 无 ST 过滤 / 幸存者偏差</td><td>同前四章</td><td>高</td><td>中</td></tr>
<tr><td><strong>D29</strong></td><td><strong>日期格式混杂（已修复）</strong></td><td><strong>bp1_features.pkl 混入 4 只股票 928 条 YYYY-MM-DD 连字符日期，字符串排序错位形成"幽灵交易日"。首版 NAV 虚高（d4_全A 曾显示 +60.2%），归一为 YYYYMMDD 后重算为 +6.2%。本报告所有数字均为清洗后口径；补票章 NAV 未受影响（cohort 口径按月分桶，混杂日期量级仅 25 条）</strong></td><td><strong>高（已修）</strong></td><td>高</td></tr>
<tr><td><strong>D30</strong></td><td><strong>组内 IC 退化</strong></td><td><strong>d4 封顶 15，信号组内截面 std=0，组内 IC 不可定义；以全宇宙日截面 IC（+0.039）为主检验</strong></td><td>中</td><td>低</td></tr>
<tr><td><strong>D31</strong></td><td><strong>~bp1_tech 过滤几乎不绑定</strong></td><td><strong>含突破对照仅多 96 条（155,371 vs 155,275）——压缩态与突破态定义互斥，伏击 vs 追入对比是两组信号的对比，不是过滤器的贡献</strong></td><td>低</td><td>低</td></tr>
<tr><td><strong>D32</strong></td><td><strong>cohort 与 NAV 口径背离</strong></td><td><strong>月度 cohort（每月等权）与日度 NAV（再平衡）在验证期方向相反（−6.9% vs +26.5% 池内）——月度信号量 263~16,948 条极度不均，cohort 同权重放大了小样本月。产品化结论以 NAV 为准，cohort 参考</strong></td><td>中</td><td>中</td></tr>
<tr><td><strong>D33</strong></td><td><strong>信号重叠（持续态重复计数）</strong></td><td><strong>run&gt;5 组（51,289 条）是同一标的连续多日重复触发，持仓窗口大幅重叠——组内"n"不是独立样本数，t 值高估。产品化按标的×压缩段去重后样本约为原始的 1/3</strong></td><td>中</td><td>中</td></tr>
<tr><td><strong>D34</strong></td><td>右截断 11.6%</td><td>20,335/175,610 条剔除（末段信号持有期超出数据截止）</td><td>中</td><td>低</td></tr>
</table>
</div>

<!-- ============ 八、验收结论 ============ -->
<h2>八、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>8.1 前视检查</h3>
<table>
<tr><th>#</th><th>检查项</th><th>结果</th></tr>
<tr><td>1</td><td>信号 T 日收盘后可知</td><td><span class="tag tag-green">通过</span> — d4 / platform_range / bp1_tech 全部 ≤ T 日</td></tr>
<tr><td>2</td><td>T+1 开盘入场 / T+20 收盘出场</td><td><span class="tag tag-green">通过</span> — t1_20 = c[T+20]/o[T+1] − 1 逐票核对</td></tr>
<tr><td>3</td><td>日期口径</td><td><span class="tag tag-green">通过（修复后）</span> — D29 连字符日期已归一</td></tr>
<tr><td>4</td><td>IC 口径</td><td><span class="tag tag-amber">组内退化已披露</span> — D30，改用全宇宙截面 IC</td></tr>
<tr><td>5</td><td>右截断</td><td><span class="tag tag-amber">20,335 条剔除（11.6%）</span> — D34</td></tr>
</table>

<h3>8.2 最终结论</h3>
<div class="warn">
<strong>产品化验收：不通过（组合层否决）。机制验证：通过（微观层全面成立）。</strong><br><br>
· <strong>信号层：通过</strong>——+2.65%/20日 vs 宇宙 +1.41%（边际 +1.24 pct），全宇宙 IC +0.039（t=+8.8），验证期边际 +0.55 pct 不消失，参数面平缓（OFAT 全部 &gt;+1.4%）<br>
· <strong>组合层：不通过</strong>——NAV d4_全A <span class="down">+6.2%</span> vs 基准 <span class="up">+31.6%</span>（跑输 25.4 pct），验证期 −8.9% 转负；月度 cohort 各档全部跑输基准 cohort<br>
· <strong>否决机制</strong>：日均 4,118 个活跃仓位（峰值 19,585）→ 组合退化为稀释 beta；每笔边际被重叠与再平衡摊薄（仓位加权 +0.153%/日 → 组合等权 +0.035%/日）<br>
· <strong>保留项</strong>：池内限定版本——NAV <span class="up">+51.3%</span>（验证期 <span class="up">+26.5%</span>，夏普 0.94），池内-验证期信号层 +3.37%/笔（n=3,883）<strong>样本外不衰减</strong>（池外验证期只剩 +0.68%）<br>
· <strong>微观机制</strong>：压缩越久越好（run&gt;5：+3.48%）、平台越窄越好（plat&lt;15%：+2.76%）、60 日翻倍不碰（−3.86%）、高波动禁入（vol20≥3%：−0.55%）——"压缩积蓄弹性"假设全面验证<br><br>
<strong>与全项目横向（20 日持有信号层均值）：</strong>bp1 追入 <span class="down">−0.29%</span> ≪ 伏击 fresh +1.91% &lt; 伏击全体 <strong>+2.65%</strong> &lt; 池内伏击 <strong>+3.28%</strong> ≈ 持续态伏击 <strong>+3.48%</strong>。<strong>bp1 的 A 级修复方向完全成立，但产品形态是"池内择时插件"，不是全A伏击策略。</strong>
</div>
</div>

<!-- ============ 九、修复路径 ============ -->
<h2>九、基于数据的修复路径（产品化怎么救）</h2>
<div class="card">
<table>
<tr><th>优先级</th><th>方向</th><th>依据</th></tr>
<tr><td><strong>S</strong></td><td><strong>池内限定伏击</strong>：只做选股池 848 只内的伏击信号（池内-验证期 +3.37%/笔、NAV 验证期 +26.5%）。池是 alpha 主源，伏击做池内择时节拍器</td><td>3.1 / 4.3</td></tr>
<tr><td><strong>A</strong></td><td><strong>降密度：限量 top-N</strong>：每日按 plat 升序（组内 IC −0.051）取前 N（如 N=20/50），只买压缩质量最高的——把 4,118 仓/日降到可操作规模，同时保留头部 alpha</td><td>4.4 / 6.2</td></tr>
<tr><td><strong>A</strong></td><td><strong>run≥6 过滤</strong>：只做持续态（+3.48% vs fresh +1.91%），砍掉 22% 样本换 +0.83 pct 均值，顺便缓解 D33 重复计数</td><td>6.2</td></tr>
<tr><td><strong>A</strong></td><td><strong>门禁：pre60 &lt; 50% + vol20 &lt; 3%</strong>：剔除接盘股（−3.86%）与高波动破坏组（−0.55%）</td><td>5.2 / 6.1</td></tr>
<tr><td>B</td><td>持有期拉长至 30 日：OFAT +4.74%（vs 20 日 +2.65%），弹性释放需要时间；或改 50% 峰值回撤移动止盈保右尾（D10 占 116% 盈亏）</td><td>3.4 / 5.1</td></tr>
<tr><td>C</td><td>与补票门禁合并实验：补票三重门禁（chg20d&gt;0 + pre60&lt;50% + dd20&gt;−15%）与伏击门禁同源（pre60），可共用一套风险预算</td><td>补票章 5.4</td></tr>
</table>
<div class="key">
<strong>五章总结论（更新）：</strong>没有任何一个信号的全A等权组合跑赢"等权持有"基准，但方向资产在增值——bp1 追突破是<strong>结构性反指</strong>（−53.2% cohort），B1/B2 <strong>无增量</strong>，补票<strong>正期望但不足额</strong>（+45.6%，三重门禁后 +52.5% vs 基准 +56.8%），d4 伏击<strong>信号真实、组合失败、池内可用</strong>（+51.3% vs 池内基准 +82.2%，验证期 +26.5%）。<strong>选股池 alpha 仍是唯一可靠的底仓</strong>，所有信号的正确产品形态都是"池内择时插件"。下一优先级：① 池内伏击 top-N 限量实验 ② bp2/搬砖同标准扫描 ③ 门禁版补票小仓位实盘对照。
</div>
</div>

<div class="muted" style="margin-top:30px;text-align:center">
d4 伏击因子（波动率压缩埋伏）深度评估报告 · S级产品化 ｜ 2026-10-20 ｜ A股量化研报复现专家团 Workflow B ｜ 配套：B1 / B2 / bp1 / 补票章
</div>

</div>
</body>
</html>"""

html = html.replace("__ROWS_GROUPS__", rows_groups)
html = html.replace("__ROWS_NAIVE__", rows_naive)
html = html.replace("__ROWS_DECILE__", rows_decile)
html = html.replace("__IC_ROWS__", ic_rows)
html = html.replace("__PORT_ROWS__", port_rows)
html = html.replace("__POOL_ROWS__", pool_rows)
html = html.replace("__NAV_ROWS__", nav_rows)
html = html.replace("__NAV_CHART__", nav_chart)
html = html.replace("__OFAT_ROWS__", ofat_rows)
html = html.replace("__OFAT_CHART__", ofat_chart)
html = html.replace("__REGIME_ROWS__", regime_rows)
html = html.replace("__CP_ROWS__", cp_rows)
html = html.replace("__Q_ROWS__", q_rows)
html = html.replace("__PATH_ROWS__", path_rows)
html = html.replace("__MAE_ROWS__", mae_rows)
html = html.replace("__MON_CHART__", mon_chart)
html = html.replace("__LAYER_CHART__", layer_chart)

out = os.path.join(HERE, "d4伏击因子深度评估报告_20261020.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("saved ->", out, "len:", len(html))
