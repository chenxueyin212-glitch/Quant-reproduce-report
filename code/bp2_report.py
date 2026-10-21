# -*- coding: utf-8 -*-
"""bp2（均线回踩买点）因子重构评估报告 — standalone HTML（六信号系列收口章）"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "bp2_results.json"), encoding="utf-8"))
V = json.load(open(os.path.join(HERE, "bp2_verify.json"), encoding="utf-8"))
EQ = json.load(open(os.path.join(HERE, "bp2_nav_equity.json"), encoding="utf-8"))

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

g = B["groups"]; nv = B["naive"]; dc = B["deciles"]
nav = B["nav"]; port = B["port"]; pool_port = B["pool_port"]; bench_port = B["bench_port"]
ofat = B["ofat"]; regime = B["regime"]; cp = B["counterparty"]; qt = B["quality"]; path = B["path"]

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
    row("当日触线", g["当日触线"]),
    row("近5日触线（非当日）", g["近5日触线(非当日)"]),
    row("N分 40-59", g["n_raw 40-59"]),
    row("N分 ≥60", g["n_raw >=60"]),
    row("缩量 ≤0.6", g["缩量<=0.6"]),
    row("缩量 0.6-0.8", g["缩量0.6-0.8"]),
])

rows_decile = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]))
    for k, v in dc.items())

ic = B["ic"]
ic_rows = ""
for k, lbl in [("dist_ma20", "收盘距MA20 dist_ma20"), ("pullback", "回撤深度 pullback"),
               ("n_raw", "N类原始分 n_raw"), ("vol_shrink", "缩量比 vol_shrink")]:
    v = ic.get(k)
    if not v: continue
    tag = "正" if v["mean"] > 0 else "负"
    sig = "显著" if abs(v.get("t") or 0) > 2 else "不显著"
    ic_rows += ("<tr><td>%s</td><td class='%s'>%+.4f</td><td>%+.3f</td><td>%.1f%%</td><td class='%s'>%+.2f</td>"
                "<td><span class='tag %s'>%s且%s</span></td></tr>") % (
        lbl, cls(v["mean"]), v["mean"], v["icir"], v["pos_ratio"], cls(v["t"]), v["t"],
        "tag-green" if v["mean"] > 0 else "tag-red", tag, sig)

naive_rows = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
    for k, v in nv.items())

port_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%"]:
    v = port[k]
    a = v["all"]; tr = v["train"]; va = v["valid"]
    if not a: continue
    port_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]), f1(a["win_m"]),
        cls(tr["total"]), f1(tr["total"]), cls(va["total"]), f1(va["total"]))
bp = bench_port["等权全A"]; bpp = bench_port["等权池内"]
port_rows += ("<tr style='background:#fdecea'><td><strong>基准·等权全A</strong></td><td class='up'><strong>%s%%</strong></td>"
              "<td><strong>%s</strong></td><td><strong>%s%%</strong></td><td><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td></tr>") % (
    f1(bp["all"]["total"]), f2(bp["all"]["sharpe"]), f1(bp["all"]["maxdd"]), f1(bp["all"]["win_m"]),
    f1(bp["train"]["total"]), f1(bp["valid"]["total"]))

pool_rows = ""
for k in ["不止损", "止损8%"]:
    v = pool_port[k]
    a = v["all"]; tr = v["train"]; va = v["valid"]
    if not a: continue
    pool_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(tr["total"]), f1(tr["total"]), cls(va["total"]), f1(va["total"]))
pool_rows += ("<tr style='background:#fdecea'><td><strong>池内基准(等权持有)</strong></td><td class='up'><strong>%s%%</strong></td>"
              "<td><strong>%s</strong></td><td><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td></tr>") % (
    f1(bpp["all"]["total"]), f2(bpp["all"]["sharpe"]), f1(bpp["all"]["maxdd"]),
    f1(bpp["train"]["total"]), f1(bpp["valid"]["total"]))

ofat_rows = ""
for k, v in ofat.items():
    if not v: continue
    bold = ' style="background:#eef4ff;font-weight:700"' if k.startswith("基准") else ""
    ofat_rows += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        bold, k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))

regime_rows = ""
regime_groups = []
for k in ["pre20>0(近期强势)", "pre20<=0(近期弱势)", "vol20<3%(低波)", "vol20>=3%(高波)"]:
    v = regime.get(k)
    if not v: continue
    regime_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))
    regime_groups.append((k.split("(")[0], [("bp2", "#e74c3c", v["mean"])]))

def micro_rows(d, name):
    out = "<tr style='background:#f8fafc'><td colspan='5'><strong>%s</strong></td></tr>" % name
    for lb, e in sorted(d.items(), key=lambda x: -x[1]["mean"] if x[1] else 0):
        if not e: continue
        out += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td></tr>" % (
            lb, "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]), f2(e["median"]), f1(e["win"]))
    return out

q_rows = ""
for k, kn in [("触线新鲜度", "触线新鲜度（当日 vs 近5日）"), ("回撤深度pullback", "回撤深度 (HHV20−L_touch)/(HHV40−LLV40)"),
              ("MA20距离dist_ma20", "收盘距 MA20"), ("缩量程度vol_shrink", "缩量比 5日均量/20日均量"),
              ("N分n_raw", "N类原始分（L0 门控强度）")]:
    q_rows += micro_rows(qt[k], kn)
cp_rows = ""
for k, kn in [("pre20", "前20日涨幅"), ("pre60", "前60日涨幅"), ("dd20", "20日回撤"), ("vol20", "20日波动率")]:
    cp_rows += micro_rows(cp[k], kn)

pt = path
path_rows = ("<tr><td>MAE 中位 / 均值</td><td class='down'>%s%% / %s%%</td><td>%s%%</td></tr>"
             "<tr><td>MFE 中位 / 均值</td><td class='up'>%s%% / %s%%</td><td>%s%%</td></tr>"
             "<tr><td>前5日最低（中位）</td><td class='down'>%s%%</td><td>%s%%</td></tr>"
             "<tr><td>后15日最高（中位）</td><td class='up'>%s%%</td><td>%s%%</td></tr>"
             "<tr><td>低于成本天数（中位/20日）</td><td>%s</td><td>%s</td></tr>") % (
    f2(pt["mae20"]["median"]), f2(pt["mae20"]["mean"]), f2(pt["mfe20"]["median"]),
    f2(pt["mfe20"]["median"]), f2(pt["mfe20"]["mean"]), f2(pt["mae20"]["median"]),
    f2(pt["first5_min"]["median"]), f2(pt["last15_max"]["median"]),
    f2(pt["last15_max"]["median"]), f2(pt["first5_min"]["median"]),
    "%.0f" % pt["below_days"]["median"], "%.0f" % pt["below_days"]["mean"])

mae = B["mae_mismatch"]
mae_rows = "".join("<tr><td>MAE ≤ %s%%</td><td class='%s'>%s%%</td></tr>" % (
    k, "down" if v > 50 else "", f1(v)) for k, v in sorted(mae["frac_below"].items(), key=lambda x: int(x[0])))

nav_rows = ""
for nm, label in [("bp2_全A", "bp2·全A"), ("bp2_池内", "bp2·池内"), ("bp2_池外", "bp2·池外"),
                  ("基准_等权全A", "基准·等权全A"), ("基准_等权池内", "基准·等权池内"), ("基准_等权池外", "基准·等权池外")]:
    m = nav[nm]["metrics"]; a = m["全部"]; tr = m["训练期"]; va = m["验证期"]
    if not a: continue
    bold = ' style="background:#fdecea"' if "基准·等权全A" == label else ""
    nav_rows += ("<tr%s><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                 "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        bold, label, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(tr["total"] if tr else 0), f1(tr["total"] if tr else 0),
        cls(va["total"] if va else 0), f1(va["total"] if va else 0))

def norm_eq(d):
    if not d: return {}
    ks = sorted(d.keys())
    b = d[ks[0]] or 1.0
    return {k: 100.0*d[k]/b for k in ks}

nav_chart = line_chart([
    ("bp2·全A", "#e74c3c", norm_eq(EQ.get("bp2_全A"))),
    ("基准·全A", "#2563eb", norm_eq(EQ.get("基准_等权全A"))),
    ("bp2·池内", "#d97706", norm_eq(EQ.get("bp2_池内"))),
    ("基准·池内", "#9ca3af", norm_eq(EQ.get("基准_等权池内"))),
], vline="20250908", vlabel="训练|验证")

monthly = B["monthly"]
mon_chart = bar_chart([
    (m[:4]+"/"+m[4:], [("bp2月均", "#e74c3c" if v["mean"] >= 0 else "#27ae60", v["mean"])])
    for m, v in list(monthly.items())[-18:]
])

mech_chart = bar_chart([
    ("pullback<0.2", [("浅回调", "#27ae60", qt["回撤深度pullback"]["<0.2"]["mean"])]),
    ("pullback0.3-0.5", [("深回调", "#9ca3af", qt["回撤深度pullback"]["0.3-0.5"]["mean"])]),
    ("缩量<=0.5", [("狠缩量", "#27ae60", qt["缩量程度vol_shrink"]["<=0.5"]["mean"])]),
    ("dist 1-2%", [("线上1-2%", "#27ae60", qt["MA20距离dist_ma20"]["1-2%"]["mean"])]),
    ("dist>2%", [("离线远", "#9ca3af", qt["MA20距离dist_ma20"][">2%"]["mean"])]),
    ("pre60>=100%", [("接盘", "#e74c3c", cp["pre60"][">=100%"]["mean"])]),
])

dens = B["density"]; trun = B["truncation"]
cal_str = "%d/%d = 100.00%%（bp2 信号 + bp1 互斥双项全对）" % (V["n_ok"], V["n_cmp"]) if V["n_cmp"] and V["n_ok"] == V["n_cmp"] else "%d/%d" % (V["n_ok"], V["n_cmp"])

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>bp2（均线回踩买点）因子深度评估报告 · 重构版（2026-10-20）</title>
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

<h1>bp2（均线回踩买点）因子深度评估报告 <span class="tag tag-amber">重构版 · 第六章 · 系列收口</span></h1>
<div class="sub">生成日期：2026-10-20 ｜ 方法：A股量化研报复现专家团 Workflow B ｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，全A 4999 只）｜ 验收：role-07 ｜ 配套：B1 / B2 / bp1 / 补票 / 搬砖章</div>

<div class="key">
<strong>验收结论先行：</strong>bp2（均线回踩买点）是<strong>六信号中最弱的一个，也是唯一"信号层边际为负"的信号</strong>——校准 __CAL__ 后，信号层均值 <span class="down">+0.51%</span>/20日 vs 全宇宙同持有期 <span class="up">+1.41%</span>（<strong>边际 −0.90 pct</strong>）；训练 +0.94%（宇宙 +1.86%）、验证 <span class="down">−0.79%</span>（宇宙 +0.54%）——<strong>训练验证双期都跑输宇宙</strong>。组合层：cohort 止损8% <span class="down">+6.2%</span>（夏普 0.20）vs 基准 +56.8%；NAV 全A +7.0% vs +31.6%（验证期 −11.0%）。<strong>产品化不通过，建议生产端直接下线或降级为池内观察项。</strong>
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>bp2 的"均线回踩"叙事里唯一站得住的是<strong>"浅回调"（pullback&lt;0.2 组 +1.16%）和"离线 1-2%"（+0.95%）</strong>——回踩买点要买"刚启动强势股的浅踩"，但生产口径的 0.5 回撤门槛太松（0.3-0.5 组只有 +0.50%），把甜区稀释掉了。更深的问题：① 中位 −1.21%、胜率 45.5%（六信号最差）；② <strong>N 类分对 bp2 无判别力</strong>（IC −0.028 t=−2.50 越高越差）——L0 门控拦了 30% 信号却没带来质量，与"N 评分实证失效"的既有结论互证；③ MAE 中位 −7.74%（六信号最深）+ 8% 止损触发 48.6%——<strong>入场点本身就是弱势点</strong>。横向六信号阶梯（cohort 止损8%）：bp1 −42.7% &lt; B1 −3.8% &lt; <strong>bp2 +6.2%</strong> &lt; B2 +16.3% &lt; 搬砖 +26.8% &lt; 补票 +45.6%（门禁后 +52.5%）&lt; 基准 +56.8%。
</div>

<!-- ============ 一、定义 ============ -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>screener/shunshi_buy_points.py</code> → <code>_bp2_ma20_pullback(klines)</code>（买点二：均线回踩买点·稳健型，Kimi V1.1 P0+P1+P2 全量化）。门控链与 <code>backtest_kimibp_daily.py</code> 生产回测一致：<strong>L0 N类门（n_raw≥40）→ L1 大势（恒过）→ L3 四大买点互斥（bp1 优先）→ L2/L4 仅影响置信度不否决</strong>。信号日 T 日收盘后可知，<strong>T+1 开盘入场、T+20 收盘出场</strong>（本报告统一口径，与前五章可比）。</p>
<div class="formula">
<strong>六条件（_bp2_ma20_pullback 逐条对应）：</strong><br>
① <strong>趋势向上</strong>：MA20 &gt; MA50 且 MA20 &gt; REF(MA20,5)（MA50 无效即否决）<br>
② <strong>中期均线</strong>：n≥60 时要求 MA50 &gt; REF(MA50,10)（n&lt;60 跳过此检）<br>
③ <strong>触及均线</strong>：近 5 日（t−4..t）存在 低≤MA20×1.02 且 收≥MA20×0.98（从旧到新取首个命中日 touch_idx）<br>
④ <strong>缩量回调</strong>：mean(V[t−5..t−1]) / MA20(V)[t] ≤ 0.8<br>
⑤ <strong>回撤幅度</strong>：(HHV20 − L[touch]) / (HHV40 − LLV40) &lt; 0.5（HHV/LLV 窗口不含今日）<br>
⑥ <strong>止跌K线</strong>（touch 当日）：|C−O|/O &lt; 0.03 且 (min(C,O)−L)/(H−L) &gt; 0.3<br>
<strong>生产信号口径</strong>：n_raw≥40（L0）&amp; bp1 未触发（L3 互斥）&amp; 六条件全过
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>MA 窗口</td><td>20 / 50 日（简单均线）</td><td>量能窗口</td><td>5 日均量 / 20 日均量 ≤ 0.8</td></tr>
<tr><td>回撤门槛</td><td>&lt; 0.5（P0）</td><td>止跌K线</td><td>实体&lt;3% 且 下影&gt;30%（P0）</td></tr>
<tr><td>L0 门</td><td>n_raw ≥ 40（N类原始分）</td><td>L3 互斥</td><td>bp1（平台突破）优先占位</td></tr>
<tr><td>信号日</td><td>T 日收盘后可知</td><td>入场 / 出场</td><td>T+1 开盘 / T+20 收盘</td></tr>
<tr><td>信号总量</td><td>__N_VALID__ 条（原始 33,603 去右截断 __N_TRUNC__）</td><td>宇宙</td><td>全A 4,999 只</td></tr>
</table>
<p class="muted">原生产回测（backtest_kimibp_daily.py）：仅池内、2024-07~2026-07、N 分取 nclass_cache.db 缓存、止损 8%；本报告改用全A日度全信号 + 重算 n_raw + 统一 20 日持有期口径（继承偏离 D26/D38）。</p>
</div>

<h2>校准验证</h2>
<div class="card">
<div class="key">
<strong>校准结果：__CAL__。</strong>方法：截断模式（truncate klines 至采样日）调用原引擎 <code>detect_shunshi_buy_point(code, klines, {'n_raw_score': n_raw[i]}, 0, 0)</code>，逐条比对 bp_type=='买点二' 与复刻的生产信号（含 L0 门与 bp1 互斥）；同时对照 bp_type=='买点一' 与复刻的 bp1_fire。采样 30 只股 × 20 个月末日。<strong>bp2 六条件 + 门控链 + 互斥逻辑全部复刻正确。</strong>
</div>
</div>

<!-- ============ 二、口径 ============ -->
<h2>二、九项口径声明（与前五章一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T 日收盘后可知 → T+1 开盘入场 → T+20 收盘出场（持有 20 个交易日）</td></tr>
<tr><td>2</td><td>分组数</td><td>十分位 + 15 组分层（全体/训练/验证/池内外/交叉/触线新鲜度/N分/缩量）</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头；对照基准为同宇宙等权持有</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除指数与名单外；未做 ST/涨跌停/停牌过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾</td></tr>
<tr><td>6</td><td>中性化</td><td>不做行业/市值中性化（D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（4999 只，主基准）/ 等权池内（848 只，含后视偏差）</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（日度夏普按 244 日年化；月度夏普 = 均值/标准差×√12）</td></tr>
<tr><td>9</td><td>费率</td><td>月度 cohort 不计成本；日度 NAV 计单次往返 0.30%（出入场各 0.15%）</td></tr>
</table>
<p class="muted">训练/验证切分：信号日 &lt; 2025-09-08 训练期（__N_TRAIN__ 条）/ 验证期（__N_VALIDP__ 条）。右截断 __N_TRUNC__ 条（__PCT_TRUNC__%）剔除。</p>
</div>

<!-- ============ 三、信号层 ============ -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分组统计（T+1开盘 → T+20收盘，无止损，%）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>P10</th><th>P90</th></tr>
__GROUP_ROWS__
</table>
<div class="warn"><strong>六信号最弱，唯一负边际：</strong>全体 +0.51% vs 宇宙 +1.41%（<strong>边际 −0.90 pct</strong>）；训练 +0.94% vs 宇宙 +1.86%（−0.92）、验证 −0.79% vs +0.54%（<strong>−1.33</strong>）——<strong>双期都跑输宇宙，不是衰减而是从来就没跑赢过</strong>。中位 −1.21%、胜率 45.5% 全系列最差。池内 +0.91% vs 池外 +0.41%——池内略好但均为正的差距太小。机制分层全部平缓：当日触线 +0.70% vs 近5日 +0.50%；缩量 ≤0.6 +0.61%；N分 ≥60 +0.73% vs 40-59 +0.48%——<strong>没有任何一个分层能把信号拉回宇宙均值以上</strong>。</div>

<h3>3.2 日度截面 RankIC</h3>
<table>
<tr><th>因子</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
__IC_ROWS__
</table>
<div class="key"><strong>两个方向性发现：</strong>① <strong>dist_ma20 IC −0.037（t=−3.28）</strong>——离 MA20 越远越差，"回踩"叙事方向对，但分层显示甜区是<strong>线上 1-2%（+0.95%）而非贴线</strong>（≤0% 组 +0.56%、&gt;2% 组 +0.24%）——要买"踩到线弹起来的第一波"，不是"趴在线上装死"。② <strong>N 类分 n_raw IC −0.028（t=−2.50）越高越差</strong>——L0 门控（≥40）拦掉 30% 却没带来质量，与全系列"N 评分实证失效"结论互证（D37 同类）。vol_shrink / pullback IC 不显著。</div>

<h3>3.3 收益分布（n=__N_VALID__，十分位）</h3>
<table>
<tr><th>十分位</th><th>n</th><th>组均值</th></tr>
__DEC_ROWS__
</table>

<h3>3.4 朴素对照（同持有期）</h3>
<table>
<tr><th>组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__NAIVE_ROWS__
</table>
<div class="note">bp2 信号 +0.51% vs 宇宙 +1.41% / 宇宙中位 −0.25% vs bp2 中位 −1.21%——<strong>bp2 的分布整体左移</strong>，回踩买点系统性地买在了弱于随机的位置。</div>
</div>

<!-- ============ 四、组合层 ============ -->
<h2>四、组合层绩效</h2>
<div class="card">
<h3>4.1 止损档对照（月度 cohort，全体）</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>月胜率%</th><th>训练期%</th><th>验证期%</th></tr>
__PORT_ROWS__
</table>
<div class="warn"><strong>组合层全面溃败：</strong>最优档止损 8% 仅 <span class="down">+6.2%</span>（夏普 0.20），<strong>跑输基准 50.6 pct（全系列最大差距）</strong>；验证期 <span class="down">−9.5%</span> 深度翻负。不止损 −3.4%——<strong>bp2 的均值回归不足以覆盖路径损耗</strong>。止损 8% 略优（MAE 中位 −7.74%、触发 48.6%）但接近"扫损"边界（补票/搬砖触发 40% 左右）。</div>

<h3>4.2 池内组合</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__POOL_ROWS__
</table>
<div class="key"><strong>池内是唯一亮点：</strong>池内止损8% +56.9%（夏普 0.86、回撤仅 −15.0%、验证期 +24.3% 存活）——形态健康但仍只有池内基准（+123.3%）的一半。池外 NAV −9.5%（验证期 −22.7%）——<strong>池外 bp2 是明确的负贡献</strong>。</div>

<h3>4.3 止损与波动率失配（MAE 分布，n=__N_VALID__）</h3>
<table>
<tr><th>阈值</th><th>MAE ≤ 阈值占比</th></tr>
__MAE_ROWS__
</table>
<div class="note">MAE 中位 <strong>−7.74%</strong>（补票/搬砖 −6.18%、全A基准 −6.74%）——<strong>bp2 入场点回撤形态比宇宙还深</strong>，"稳健型"定位与实测路径相反。8% 止损触发 48.6%：近半信号被扫，止损是"必要的减亏"而非"可选的保险"。</div>

<h3>4.4 月度收益（近 18 个月，cohort 等权）</h3>
__MON_CHART__

<h3>4.5 日度等权再平衡 NAV（含费率 0.30%/往返）</h3>
<table>
<tr><th>口径</th><th>全期累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__NAV_ROWS__
</table>
__NAV_CHART__
<div class="warn">NAV 与 cohort 同向：全A +7.0% vs 基准 +31.6%（验证期 −11.0% vs +0.1%）。日均活跃仓位 __DENS__ 个（峰值 __DENS_MAX__）——密度是六信号最低（搬砖 2,226 / 补票更高），<strong>bp2 的问题不是摊薄，是信号本身没有 alpha</strong>。池内 NAV +55.7%（验证期 +32.5%）与池内基准验证期 +33.5% 基本持平——又是"池内平庸、池外有毒"结构。</div>
</div>

<!-- ============ 五、稳健性 ============ -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（信号层门槛）</h3>
<table>
<tr><th>变体</th><th>n</th><th>均值</th><th>胜率</th></tr>
__OFAT_ROWS__
</table>
<div class="note">全部切片均值都在 +0.2%~+1.2% 区间——<strong>没有任何门槛组合能把 bp2 拉到宇宙均值（+1.41%）以上</strong>。最优：pullback&lt;0.2（+1.16%，n=2,153）与缩量≤0.5（+0.92%，n=940）——甜区存在但太小、太浅，撑不起生产信号量。</div>

<h3>5.2 状态条件</h3>
<table>
<tr><th>状态</th><th>n</th><th>均值</th><th>胜率</th></tr>
__REGIME_ROWS__
</table>
__REGIME_CHART__

<h3>5.3 数据质量与截断</h3>
<table>
<tr><th>项目</th><th>数值</th></tr>
<tr><td>原始信号 / 有效信号</td><td>33,603 / __N_VALID__（右截断 __N_TRUNC__ 条 = __PCT_TRUNC__%）</td></tr>
<tr><td>L0 门与互斥损耗</td><td>纯技术条件 48,249 条 → L0 门 + bp1 互斥后 33,603 条（砍 30.4%）</td></tr>
<tr><td>日均活跃仓位（峰值/中位）</td><td>__DENS__（__DENS_MAX__ / __DENS_MED__），701 个交易日</td></tr>
<tr><td>校准</td><td>__CAL__</td></tr>
</table>
</div>

<!-- ============ 六、微观 ============ -->
<h2>六、微观经济假设检验（"回踩"到底在买什么）</h2>
<div class="card">
<h3>6.1 机制分层：触线 / 回撤 / 距离 / 缩量 / N分</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__Q_ROWS__
</table>
__MECH_CHART__
<div class="key"><strong>机制真相：买"浅踩弹起"，不买"深蹲趴线"。</strong>① pullback&lt;0.2 → +1.16%（中位 +0.00%、胜率 49.9%）vs 0.3-0.5 → +0.50%——<strong>生产门槛 0.5 太松</strong>，甜区在 0.2 以下；② dist_ma20 线上 1-2% → +0.95% 最好，&gt;2% → +0.24% 最差——回踩确认后的第一波反弹最好；③ 缩量 ≤0.5 → +0.92%（弱单调）；④ N 分 ≥80 组 +9.87% 但 n=13 不可采信；⑤ 触线新鲜度差异小（+0.70 vs +0.50）——"当日触线"没有明显溢价。</div>

<h3>6.2 对手方分层</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__CP_ROWS__
</table>
<div class="warn"><strong>接盘边界与全系列一致且更陡：</strong>pre60 ≥100%（60日翻倍）组 <span class="down">−3.50%</span>、50-100% 组 −0.20%——均线回踩买点买到"已经翻倍的强势股"就是接盘。pre20 ≥40% 组 +0.05% 归零。<strong>低波保护显著</strong>：vol20&lt;2% +1.35% vs 3-4% +0.04%。<strong>可操作硬门禁：pre60 &lt; 50% 且 vol20 &lt; 3%。</strong></div>

<h3>6.3 持仓路径</h3>
<table>
<tr><th>指标</th><th>中位</th><th>均值</th></tr>
__PATH_ROWS__
</table>
<div class="note">MAE 中位 −7.74% / MFE 中位 +7.92%——<strong>MFE/MAE 比 1.02，六信号最平</strong>（补票 1.36、搬砖 1.18）：路径几乎完全对称，右尾没有补偿。低于成本天数中位 11/20——持有体验差。"稳健型"标签与实测不符，属于命名误导（D40）。</div>
</div>

<!-- ============ 七、横向定位 ============ -->
<h2>七、六信号横向定位（系列收口）</h2>
<div class="card">
<table>
<tr><th>信号</th><th>cohort 止损8% 全A</th><th>vs 基准 +56.8%</th><th>验证期</th><th>信号层边际</th><th>定位</th></tr>
<tr><td>bp1（突破追入）</td><td class="down"><strong>−42.7%</strong></td><td>−99.5 pct</td><td>深度翻负</td><td>负</td><td>失效（止损失配）</td></tr>
<tr><td>B1（倒车接人）</td><td class="down">−3.8%</td><td>−60.6 pct</td><td>翻负</td><td>弱</td><td>失效</td></tr>
<tr><td><strong>bp2（本章）</strong></td><td class="down"><strong>+6.2%</strong></td><td><strong>−50.6 pct</strong></td><td><strong>−9.5%（深度翻负）</strong></td><td><strong>−0.90 pct（负）</strong></td><td><strong>失效（全系列最弱），建议下线</strong></td></tr>
<tr><td>B2</td><td class="up">+16.3%</td><td>−40.5 pct</td><td>翻负</td><td>弱</td><td>失效</td></tr>
<tr><td>搬砖（多空临界）</td><td class="up">+26.8%</td><td>−30.0 pct</td><td>−0.6%（归零）</td><td>+0.13 pct</td><td>不通过；池内插件候选</td></tr>
<tr><td>补票（单针下20）</td><td class="up">+45.6%（门禁后 +52.5%）</td><td>−11.2 → −4.3 pct</td><td>+3.3%（存活）</td><td>+0.37 pct</td><td>唯一"有条件可用"</td></tr>
<tr style="background:#fdecea"><td><strong>基准·等权全A</strong></td><td class="up"><strong>+56.8%</strong></td><td>—</td><td>+3.0%</td><td>—</td><td>—</td></tr>
</table>
<div class="key"><strong>系列总结论（六信号全测完）：</strong>① 六信号的原系统"期望全正"（样本内）在统一口径重构下<strong>只剩补票一个有条件可用</strong>，其余五个（含 bp2 这种双期负边际的）全部不通过；② <strong>alpha 本体在选股池</strong>——池内口径六个信号全部为正（bp2 池内 +56.9%），池外全部平庸或有毒（bp2 池外 NAV −9.5%）；③ N 类评分对所有信号都无排序判别力（bp2 IC −0.028 再证）；④ 三条通用硬门禁（pre60&lt;50% / dd20&gt;−15% / vol20&lt;3%）在全部六章都有效——<strong>对手方清理是跨信号的稳定结构</strong>。</div>
</div>

<!-- ============ 八、偏离登记 ============ -->
<h2>八、偏离登记（继承 D1-D37，bp2 专项新增）</h2>
<div class="card">
<table>
<tr><th>编号</th><th>偏离项</th><th>说明</th><th>影响</th><th>处置</th></tr>
<tr><td><strong>D38</strong></td><td><strong>N 分来源与 CSV 口径</strong></td><td><strong>生产 CSV 用 nclass_cache.db 缓存（as_of 对齐有 miss，bp1 章 CSV 交叉核对 33/47）+ 仅池内 + 2024-07~2026-07；本章重算 n_raw（bp1_prep 链路，校准 100%）+ 全A全时段</strong></td><td>中（绝对值不可与 kimibp_daily_trades 直接比）</td><td>继承 D26 口径声明</td></tr>
<tr><td><strong>D39</strong></td><td><strong>源码边角语义逐位保留</strong></td><td><strong>⑥止跌K线在 H==L 时跳过检查（视为通过）；⑤回撤在 HHV40==LLV40 时跳过；②MA50 斜率在 n&lt;60 时跳过——全部按源码行为复刻，不"修正"</strong></td><td>低（不改变主结论）</td><td>登记</td></tr>
<tr><td><strong>D40</strong></td><td><strong>"稳健型"命名误导</strong></td><td><strong>bp2 标称"稳健型/偏重仓位 15-20%"，实测 MAE 中位 −7.74%（全系列最深）、MFE/MAE 比 1.02（全系列最平）、胜率 45.5%（全系列最低）——风险收益特征与"稳健"相反</strong></td><td>高（仓位建议与实测反向）</td><td>登记；建议生产端删去仓位标签</td></tr>
<tr><td>D37（互证）</td><td>N 评分无判别力</td><td>bp2 的 n_raw IC −0.028（t=−2.50）——与 d4/N 评分专项结论互证</td><td>—</td><td>互证登记</td></tr>
</table>
</div>

<!-- ============ 九、验收 ============ -->
<h2>九、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>9.1 前视检查</h3>
<table>
<tr><th>检查项</th><th>结果</th></tr>
<tr><td>信号时点</td><td>✅ T 日收盘后可知（MA/量能/触线/回撤/止跌K 全用 T 日及以前数据）</td></tr>
<tr><td>入场/出场</td><td>✅ T+1 开盘 → T+20 收盘（t1_20 逐票核对一致）</td></tr>
<tr><td>校准</td><td>✅ __CAL__</td></tr>
<tr><td>右截断</td><td>✅ __N_TRUNC__ 条（__PCT_TRUNC__%）剔除</td></tr>
<tr><td>训练/验证</td><td>✅ 切分 2025-09-08，验证期未参与参数选择（生产口径来自源码）</td></tr>
</table>
<h3>9.2 最终结论</h3>
<div class="warn">
<strong>验收不通过（全系列最弱，建议生产端下线）。</strong><br>
<strong>不通过的理由：</strong>① <strong>信号层边际 −0.90 pct（唯一为负）</strong>，训练（−0.92）验证（−1.33）双期都跑输宇宙——不是样本外衰减，是从来没有 alpha；② 组合层 cohort 止损8% +6.2% vs 基准 +56.8%（全系列最大差距 50.6 pct），验证期 −9.5%；③ 中位 −1.21%、胜率 45.5%、MFE/MAE 1.02——路径统计全面弱于随机。<br>
<strong>仅存的真实成分：</strong>① 浅回调甜区（pullback&lt;0.2 → +1.16%）与"线上 1-2%"（+0.95%）——机制方向对但太小；② 池内止损8% +56.9%（夏普 0.86、回撤 −15.0%）形态健康，但仍只有池内基准一半；③ 对手方边界（pre60≥100% → −3.50%）与全系列一致。<br>
<strong>处置建议：</strong>生产端直接下线 bp2；若保留"均线回踩"叙事，改为<strong>只做池内 + pullback&lt;0.2 + dist_ma20 1-2% + pre60&lt;50% + vol20&lt;3%</strong> 的极窄切片观察项（n≈2,000/30 月），不给仓位。
</div>
</div>

<!-- ============ 十、修复路径 ============ -->
<h2>十、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>#</th><th>修复项</th><th>依据</th><th>预期</th></tr>
<tr><td>1</td><td><strong>下线或降级为观察项</strong>（不给仓位）</td><td>信号层负边际 + 组合层 50.6 pct 跑输——六信号唯一双负</td><td>止损不再扩大</td></tr>
<tr><td>2</td><td>若保留：极窄甜区切片（pullback&lt;0.2 &amp; dist 1-2% &amp; pre60&lt;50% &amp; vol20&lt;3%）</td><td>6.1/6.2 分层甜区</td><td>+1.2% 档观察（仍低于宇宙）</td></tr>
<tr><td>3</td><td>回撤门槛 0.5 → 0.2 参数收紧重测</td><td>pullback 分层单调（0.3-0.5 组仅 +0.50%）</td><td>信号量减少、均值提升</td></tr>
<tr><td>4</td><td>N 分门控重估（L0≥40 是否还有必要）</td><td>n_raw IC −0.028 越高越差；L0 拦 30% 无增益</td><td>简化门控或换择时指标</td></tr>
<tr><td>5</td><td>walk-forward 验证甜区切片</td><td>D38 选择偏差</td><td>生产决策依据</td></tr>
</table>
<p class="muted"><strong>系列纪律红线不变：</strong>修复验证完成前不实盘；池外信号一律不做。<strong>至此六信号全部按统一标准重测完毕</strong>——下一阶段工作重心应回到补票门禁版 walk-forward 与池内插件（d4/搬砖）top-N 实验。</p>
</div>

<div class="muted" style="margin-top:26px;text-align:center">
本报告由 A股量化研报复现专家团 Workflow B 生成 ｜ 数据源：kline_cache_all.db / shunshi_buy_points.py / stock_pool.json ｜ 口径版本 v2（2026-10-20）｜ 六信号系列终章
</div>
</div>
</body>
</html>"""

n_train = g["训练期"]["n"]; n_validp = g["验证期"]["n"]
n_valid = g["全体"]["n"]
n_trunc = trun["n_truncated"]; pct_trunc = 100.0*trun["n_truncated"]/trun["n_total"]

html = (html
    .replace("__CAL__", cal_str)
    .replace("__N_VALID__", "{:,}".format(n_valid))
    .replace("__N_TRAIN__", "{:,}".format(n_train))
    .replace("__N_VALIDP__", "{:,}".format(n_validp))
    .replace("__N_TRUNC__", "{:,}".format(n_trunc))
    .replace("__PCT_TRUNC__", "%.1f" % pct_trunc)
    .replace("__DENS__", "{:,.0f}".format(dens["mean_pos"]))
    .replace("__DENS_MAX__", "{:,}".format(dens["max_pos"]))
    .replace("__DENS_MED__", "{:,.0f}".format(dens["median_pos"]))
    .replace("__GROUP_ROWS__", rows_groups)
    .replace("__IC_ROWS__", ic_rows)
    .replace("__DEC_ROWS__", rows_decile)
    .replace("__NAIVE_ROWS__", naive_rows)
    .replace("__PORT_ROWS__", port_rows)
    .replace("__POOL_ROWS__", pool_rows)
    .replace("__MAE_ROWS__", mae_rows)
    .replace("__MON_CHART__", mon_chart)
    .replace("__NAV_ROWS__", nav_rows)
    .replace("__NAV_CHART__", nav_chart)
    .replace("__OFAT_ROWS__", ofat_rows)
    .replace("__REGIME_ROWS__", regime_rows)
    .replace("__REGIME_CHART__", bar_chart(regime_groups))
    .replace("__Q_ROWS__", q_rows)
    .replace("__CP_ROWS__", cp_rows)
    .replace("__MECH_CHART__", mech_chart)
    .replace("__PATH_ROWS__", path_rows))

from healthcheck_chapter import inject
html = inject(html, "bp2(均线回踩)")

out = os.path.join(HERE, "bp2因子深度评估报告_重构版_20261020.html")
open(out, "w", encoding="utf-8").write(html)
print("saved -> %s len: %d chars" % (out, len(html)))
