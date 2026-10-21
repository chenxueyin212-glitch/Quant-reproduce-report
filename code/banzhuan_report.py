# -*- coding: utf-8 -*-
"""搬砖（砖型图VAR6A·多空临界）因子重构评估报告 — standalone HTML"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "banzhuan_results.json"), encoding="utf-8"))
V = json.load(open(os.path.join(HERE, "banzhuan_verify.json"), encoding="utf-8"))
EQ = json.load(open(os.path.join(HERE, "banzhuan_nav_equity.json"), encoding="utf-8"))

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
    row("第1红砖", g["第1砖"]),
    row("第2红砖", g["第2砖"]),
    row("score = 5-6", g["score=5-6"]),
    row("score = 7", g["score=7"]),
    row("score ≥ 8", g["score>=8"]),
    row("阳线（C>O）", g["阳线(C>O)"]),
    row("阴线", g["阴线"]),
    row("站上 MA60", g["站上MA60"]),
])

rows_decile = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]))
    for k, v in dc.items())

ic = B["ic"]
ic_rows = ""
for k, lbl in [("dist_close", "收盘距多空线 dist_close"), ("dist_low", "最低距多空线 dist_low"),
               ("score", "搬砖 score（连续）"), ("chg20d", "近20日涨幅 chg20d"), ("var6a", "VAR6A 值")]:
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
    regime_groups.append((k.split("(")[0], [("搬砖", "#e74c3c", v["mean"])]))

def micro_rows(d, name):
    out = "<tr style='background:#f8fafc'><td colspan='5'><strong>%s</strong></td></tr>" % name
    for lb, e in sorted(d.items(), key=lambda x: -x[1]["mean"] if x[1] else 0):
        if not e: continue
        out += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td></tr>" % (
            lb, "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]), f2(e["median"]), f1(e["win"]))
    return out

q_rows = ""
for k, kn in [("砖位", "红砖位（第1砖 vs 第2砖）"), ("支撑距离dist_low", "支撑距离（盘中最低 − 多空线）/|多空线|"),
              ("VAR6A水平", "VAR6A 砖型强度"), ("chg20d", "近20日涨幅")]:
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

# NAV 表与曲线
nav_rows = ""
for nm, label in [("bz_全A", "搬砖·全A"), ("bz_池内", "搬砖·池内"), ("bz_池外", "搬砖·池外"),
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
    ("搬砖·全A", "#e74c3c", norm_eq(EQ.get("bz_全A"))),
    ("基准·全A", "#2563eb", norm_eq(EQ.get("基准_等权全A"))),
    ("搬砖·池内", "#d97706", norm_eq(EQ.get("bz_池内"))),
    ("基准·池内", "#9ca3af", norm_eq(EQ.get("基准_等权池内"))),
], vline="20250908", vlabel="训练|验证")

monthly = B["monthly"]
mon_chart = bar_chart([
    (m[:4]+"/"+m[4:], [("搬砖月均", "#e74c3c" if v["mean"] >= 0 else "#27ae60", v["mean"])])
    for m, v in list(monthly.items())[-18:]
])

mech_chart = bar_chart([
    ("第1砖", [("第1砖", "#e74c3c", g["第1砖"]["mean"])]),
    ("第2砖", [("第2砖", "#e74c3c", g["第2砖"]["mean"])]),
    ("VAR6A>=60", [("强砖", "#27ae60", qt["VAR6A水平"][">=60"]["mean"])]),
    ("VAR6A 20-40", [("弱砖", "#9ca3af", qt["VAR6A水平"]["20-40"]["mean"])]),
    ("chg20d 5-10%", [("中涨", "#27ae60", qt["chg20d"]["5-10%"]["mean"])]),
    ("chg20d>=10%", [("暴涨", "#9ca3af", qt["chg20d"][">=10%"]["mean"])]),
])

dens = B["density"]; trun = B["truncation"]

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>搬砖（砖型图VAR6A·多空临界）因子深度评估报告 · 重构版（2026-10-20）</title>
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

<h1>搬砖（砖型图VAR6A·多空临界）因子深度评估报告 <span class="tag tag-amber">重构版 · 第五章</span></h1>
<div class="sub">生成日期：2026-10-20 ｜ 方法：A股量化研报复现专家团 Workflow B ｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，全A 4999 只）｜ 验收：role-07 ｜ 配套：B1 / B2 / bp1 / 补票章</div>

<div class="key">
<strong>验收结论先行：</strong>搬砖（砖型图VAR6A·多空临界）<strong>产品化不通过</strong>——校准 __CAL__ 后，信号层均值 <span class="up">+1.54%</span>/20日 vs 全宇宙同持有期 <span class="up">+1.41%</span>，<strong>边际仅 +0.13 pct（五信号中最低）</strong>；训练期 +2.27% → 验证期 +0.35%，而宇宙自身验证期 +0.54%——<strong>验证期边际 −0.19 pct 翻负，样本外优势不存在</strong>。组合层：日度 NAV 全A <span class="down">+7.9%</span> vs 基准 <span class="up">+31.6%</span>；月度 cohort 止损8% +26.8%（夏普 0.50）vs 基准 +56.8%。横向（cohort 止损8%）：B1 −3.8% &lt; B2 +16.3% &lt; <strong>搬砖 +26.8%</strong> &lt; 补票 +45.6%（门禁后 +52.5%）&lt; 基准 +56.8%。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>搬砖的"多空临界"机制核心<strong>是真的</strong>——收盘贴多空线越近未来越好（dist_close IC <span class="up">−0.068</span>，t=−8.57，全章最强因子），VAR6A 强砖单调有效（≥60 → +1.62% vs 20-40 → −1.45%）。但两个致命伤：<strong>① score 评分体系方向反了</strong>（score 越高越差，IC −0.028 t=−3.59——"站上MA60"加分项被数据否定，+1.41% 低于全体）；<strong>② 信号太密</strong>（日均 2,226 仓、峰值 4,807）+ 边际太薄（+0.13 pct），组合层摊薄成 beta 影子。<strong>搬砖不是独立 alpha，是"强势股回调至多空线"的入场时机规则</strong>——池外 −3.8% vs 池内 +55.6%（NAV），全部价值在池内。最优切片：chg20d 5-10% + VAR6A≥60（+2.23%）。
</div>

<!-- ============ 一、定义 ============ -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>screener/zge_compute.py</code> → <code>check_banzhuan(ind)</code>（搬砖战法：砖型图 VAR6A 波段买卖，"第2块红砖开盘买、红转绿清仓"，属顺势回调系统）。指标由 <code>calc_zge_indicators(klines)</code> 计算。信号日 T 日收盘后可知，<strong>T+1 开盘入场、T+20 收盘出场</strong>（本报告统一口径，与 B1/B2/bp1/补票可比）。</p>
<div class="formula">
<strong>砖型图 VAR6A（搬砖核心）：</strong><br>
· X = (C − LLV(L,4)) / (HHV(H,4) − LLV(L,4)) × 100　· Y = (HHV(H,4) − C) / (HHV(H,4) − LLV(L,4)) × 100 − 90<br>
· VAR6A = SMA(SMA(X,6,1),6,1) − SMA(Y,4,1)　← 注释称 SMA(N,1)≈EMA α=1/N，<strong>实现用 α=2/(N+1)</strong>（按代码复刻，D35）<br>
· <strong>红砖</strong>：VAR6A<sub>t</sub> &gt; VAR6A<sub>t−4</sub>；绿砖：&lt;（相等日两端都不算，断连）<br><br>
<strong>评分规则（check_banzhuan 逐条对应，2026-07-17 规则变更后）：</strong><br>
① <strong>第1红砖</strong> +5 分；<strong>第2红砖</strong> +3 分；第3砖起弃用（不作为买入信号）<br>
② 趋势确认：C &gt; MA60 → +1 分<br>
③ <strong>支撑过滤（仅第1/2砖）</strong>：知行多空线 zd = (MA14+MA28+MA57+MA114)/4，touch = 低 ≤ zd×1.001 且 recovery = 收 ≥ zd×0.99 → 记"搬砖-多空临界" +2 分；阳线（C&gt;O）再 +1；<strong>支撑未确认 → score 清零</strong><br>
<strong>生产信号口径（backtest_bupiao_banzhuan.py）：score ≥ 5 且 support_type == "搬砖-多空临界"</strong>
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>砖型窗口</td><td>4 日（HHV(H,4)/LLV(L,4)）</td><td>砖比较间隔</td><td>4 期（var6a[t] vs var6a[t-4]）</td></tr>
<tr><td>第1砖</td><td>+5 分（最佳入场）</td><td>第2砖</td><td>+3 分（确认）</td></tr>
<tr><td>支撑线</td><td>知行多空线 (MA14+MA28+MA57+MA114)/4</td><td>touch/recovery</td><td>低≤线×1.001 且 收≥线×0.99</td></tr>
<tr><td>信号日</td><td>T 日收盘后可知</td><td>入场 / 出场</td><td>T+1 开盘 / T+20 收盘</td></tr>
<tr><td>信号总量</td><td>__N_VALID__ 条（完整前瞻，原始 87,263 去右截断 __N_TRUNC__）</td><td>宇宙</td><td>全A 4,999 只</td></tr>
</table>
<p class="muted">原回测（backtest_bupiao_banzhuan.py）为月频采样 + 8% 止损 + 50% 峰值回撤止盈、仅池内 848 只、补票搬砖混跑——本报告改用全A日度全信号 + 统一 20 日持有期口径，跨章可比（继承偏离 D26）。</p>
</div>

<h2>校准验证</h2>
<div class="card">
<div class="key">
<strong>校准结果：__CAL__ 全维度一致。</strong>方法：截断模式（truncate klines 至采样日）现算 vs 原引擎 <code>calc_zge_indicators</code> + <code>check_banzhuan</code> 逐条比对 score 与信号标志。采样 30 只股 × 20 个月末日。含 score 加减分（砖位/MA60/支撑/阳线）与生产信号布尔的双重比对——<strong>score 逐位一致即证明六条加减分规则全部复刻正确</strong>。
</div>
</div>

<!-- ============ 二、口径 ============ -->
<h2>二、九项口径声明（与前四章一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T 日收盘后可知 → T+1 开盘入场 → T+20 收盘出场（持有 20 个交易日）</td></tr>
<tr><td>2</td><td>分组数</td><td>十分位 + 17 组分层（全体/训练/验证/池内外/交叉/砖位/score 三档/阳线/MA60）</td></tr>
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
<div class="warn"><strong>边际过薄且样本外翻负：</strong>全体 +1.54% vs 宇宙 +1.41%（<strong>边际 +0.13 pct</strong>，补票 +0.37、d4 +1.24）。训练 +2.27%（宇宙 +1.86%，边际 +0.41）→ 验证 +0.35%（宇宙 +0.54%，<strong>边际 −0.19 pct</strong>）。中位 −0.21% 负、胜率 48.9%——均值全靠右尾。<strong>池内 +1.65% vs 池外 +1.52%：搬砖几乎不携带选股池 alpha</strong>（五信号独一份，补票池内 +2.52 vs 池外 +0.71）。砖位：第2砖 +1.68% 微好于第1砖 +1.45%——"第2砖确认"叙事微弱成立。score 分层不单调（7→+1.85% 最高、≥8→+1.44%）——评分体系无排序判别力。</div>

<h3>3.2 日度截面 RankIC</h3>
<table>
<tr><th>因子</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
__IC_ROWS__
</table>
<div class="key"><strong>机制核心成立（全章最强因子）：dist_close IC = −0.068（t = −8.57）</strong>——收盘价距多空线越远未来越差，<strong>"贴着多空线买"是搬砖唯一真正有效的成分</strong>。但 score 总分 IC −0.028（t=−3.59）<strong>方向反了</strong>：加分项"站上 MA60"被数据否定（站上 MA60 组 +1.41% 低于全体 +1.54%）——<strong>评分只能当门槛用，不能当排序用（D37）</strong>。chg20d IC −0.017（t=−1.98）：近 20 日涨太多的砖不好。</div>

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
<div class="note">全宇宙同持有期均值 +1.41%（训练 +1.86% / 验证 +0.54%）——市场本身验证期在衰减，但搬砖衰减得更狠（+2.27→+0.35），<strong>验证期跑输宇宙 −0.19 pct</strong>。</div>
</div>

<!-- ============ 四、组合层 ============ -->
<h2>四、组合层绩效</h2>
<div class="card">
<h3>4.1 止损档对照（月度 cohort，全体）</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>月胜率%</th><th>训练期%</th><th>验证期%</th></tr>
__PORT_ROWS__
</table>
<div class="warn"><strong>组合层否决：</strong>止损 8% 最优档 +26.8%（夏普 0.50、回撤 −22.5%），<strong>仍跑输基准 30.0 pct</strong>；验证期 <span class="down">−0.6%</span> 归零。止损 8% &gt; 不止损（+19.2%）——MAE 中位 −6.18%、8% 触发率 40.4%，<strong>8% 止损切左尾有效</strong>（与补票同构）。但 12%/15% 档快速恶化（+16.7%/+12.6%）——搬砖路径波动更依赖紧止损。</div>

<h3>4.2 池内组合</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__POOL_ROWS__
</table>
<div class="key"><strong>价值全在池内：</strong>池内止损8% +66.7%（夏普 0.93、回撤 −23.9%），验证期 +27.4%——<strong>样本外存活</strong>。但池内基准 +123.3% 仍大幅领先。池外 NAV −3.8%（验证期 −15.6%）——<strong>池外的搬砖是负贡献</strong>，与信号层"池内外几乎无差"（+1.65 vs +1.52）形成对照：均值相近但池外右尾更薄、组合复利更差。</div>

<h3>4.3 止损与波动率失配（MAE 分布，n=__N_VALID__）</h3>
<table>
<tr><th>阈值</th><th>MAE ≤ 阈值占比</th></tr>
__MAE_ROWS__
</table>
<div class="note">MAE 中位 −6.18% / 均值 −8.21%——与补票（中位 −6.18%）同构，入场回撤形态接近全宇宙中位（−6.74%）。8% 止损触发 40.4%：切左尾而不过度扫损，是搬砖为数不多的健康特征。</div>

<h3>4.4 月度收益（近 18 个月，cohort 等权）</h3>
__MON_CHART__

<h3>4.5 日度等权再平衡 NAV（含费率 0.30%/往返）</h3>
<table>
<tr><th>口径</th><th>全期累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__NAV_ROWS__
</table>
__NAV_CHART__
<div class="warn"><strong>NAV 口径与 cohort 方向一致（无 D32 背离）：</strong>全A NAV +7.9% vs 基准 +31.6%（验证期 −7.7% vs +0.1%）。归因：日均活跃仓位 __DENS__ 个（峰值 __DENS_MAX__）——等权组合摊薄成 beta 影子。唯一亮点：<strong>池内 NAV +55.6%，验证期 +34.4% vs 池内基准验证期 +33.5%（跑赢 +0.9 pct）</strong>——池内搬砖样本外与基准持平，全A 跑输全因池外拖累。</div>
</div>

<!-- ============ 五、稳健性 ============ -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（信号层门槛）</h3>
<table>
<tr><th>变体</th><th>n</th><th>均值</th><th>胜率</th></tr>
__OFAT_ROWS__
</table>
<div class="note">OFAT 全部平缓：score 门槛无增量（≥7 → +1.55%、≥8 → +1.44%）；<strong>chg20d&gt;0 +1.78% vs ≤0 +1.22%</strong>——顺势砖更好（但 IC 说 chg20d 连续值越高越差：中等涨幅最好，暴涨最差）；深探支撑（dist_low≤−3%）+1.69% vs 贴线 +1.48%——盘中刺破更深反而微好（承接盘更强）。</div>

<h3>5.2 状态条件</h3>
<table>
<tr><th>状态</th><th>n</th><th>均值</th><th>胜率</th></tr>
__REGIME_ROWS__
</table>
__REGIME_CHART__

<h3>5.3 数据质量与截断</h3>
<table>
<tr><th>项目</th><th>数值</th></tr>
<tr><td>原始信号 / 有效信号</td><td>87,263 / __N_VALID__（右截断 __N_TRUNC__ 条 = __PCT_TRUNC__%）</td></tr>
<tr><td>日均活跃仓位（峰值/中位）</td><td>__DENS__（__DENS_MAX__ / __DENS_MED__），708 个交易日</td></tr>
<tr><td>日期格式</td><td>D29 混杂已在 prep 归一（replace("-","")）</td></tr>
<tr><td>校准</td><td>__CAL__</td></tr>
</table>
<p class="muted">搬砖信号分布比补票均匀（78,783 条/708 天 ≈ 111 条/日），但持仓重叠使日均活跃仓位 2,226 个——距离可操作规模（≤50）仍差 45 倍。</p>
</div>

<!-- ============ 六、微观 ============ -->
<h2>六、微观经济假设检验（"多空临界"到底在买什么）</h2>
<div class="card">
<h3>6.1 机制分层：砖位 / 支撑距离 / VAR6A / 动量</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__Q_ROWS__
</table>
__MECH_CHART__
<div class="key"><strong>三条机制结论：</strong>① <strong>VAR6A 强度单调</strong>——≥60 组 +1.62%（n=73,380，占 93%）vs 20-40 组 −1.45%：砖型动能是真实成分；② <strong>chg20d 倒 U 型</strong>——5-10% 组 +2.23% 最优、≥10% 组 +0.37% 最差："刚启动的强势股回调"最好，暴涨股的砖是出货砖；③ 砖位差异小（第2砖 +1.68% vs 第1砖 +1.45%），支撑距离非单调（深探 −3% 以下 +1.69%）——<strong>"精确踩线"不比"深探回升"好</strong>，recovery 条件（收≥线×0.99）已把无效刺破筛掉。</div>

<h3>6.2 对手方分层</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__CP_ROWS__
</table>
<div class="warn"><strong>接盘边界与 B2/补票完全一致：</strong>pre20 20-40% 组 <span class="down">−1.80%</span>、≥40% 组 <span class="down">−6.70%</span>——追已大涨的砖是接盘；pre60 20-50% 组 −0.56%、≥100% 组 −1.57%。<strong>但搬砖独有的反直觉：dd20 ≥−5%（浅回撤）组 +2.14% 最好、−15~−10% 组 +0.50% 最差</strong>——搬砖要买"强势股的浅回调到线"，不是"深跌股的反抽"。vol20&lt;2%（低波）+1.93% &gt; 高波 +1.48%。<strong>可操作硬门禁（与补票同构）：pre20 &lt; 20% 且 dd20 &gt; −10%。</strong></div>

<h3>6.3 持仓路径</h3>
<table>
<tr><th>指标</th><th>中位</th><th>均值</th></tr>
__PATH_ROWS__
</table>
<div class="note">MAE 中位 −6.18% / MFE 中位 +7.32% / 前 5 日最低 −1.75% / 后 15 日最高 +4.66%——低于成本天数中位 10/20（补票 9/20）。MFE/MAE 比 1.18：路径不对称性弱于补票（1.36），右尾更薄。</div>
</div>

<!-- ============ 七、横向定位 ============ -->
<h2>七、横向定位（五信号阶梯）</h2>
<div class="card">
<table>
<tr><th>信号</th><th>cohort 止损8% 全A</th><th>vs 基准 +56.8%</th><th>验证期</th><th>信号层边际</th><th>定位</th></tr>
<tr><td>bp1（突破追入）</td><td class="down"><strong>−42.7%</strong></td><td>−99.5 pct</td><td>深度翻负</td><td>负</td><td>失效（止损失配）</td></tr>
<tr><td>B1（倒车接人）</td><td class="down">−3.8%</td><td>−60.6 pct</td><td>翻负</td><td>弱</td><td>失效</td></tr>
<tr><td>B2</td><td class="up">+16.3%</td><td>−40.5 pct</td><td>翻负</td><td>弱</td><td>失效</td></tr>
<tr><td><strong>搬砖（本章）</strong></td><td class="up"><strong>+26.8%</strong></td><td><strong>−30.0 pct</strong></td><td><strong>−0.6%（归零）</strong></td><td><strong>+0.13 pct（翻负）</strong></td><td><strong>不通过；池内插件候选（弱于 d4）</strong></td></tr>
<tr><td>补票（单针下20）</td><td class="up">+45.6%（门禁后 +52.5%）</td><td>−11.2 → −4.3 pct</td><td>+3.3%（存活）</td><td>+0.37 pct</td><td>唯一"有条件可用"</td></tr>
<tr style="background:#fdecea"><td><strong>基准·等权全A</strong></td><td class="up"><strong>+56.8%</strong></td><td>—</td><td>+3.0%</td><td>—</td><td>—</td></tr>
</table>
<p class="muted">NAV 口径补充：搬砖 全A +7.9% vs d4 全A +6.2% vs 基准 +31.6%——d4 与搬砖同属"信号密度过高摊薄"型，但 d4 信号层边际（+1.24 pct）是搬砖（+0.13）的近十倍。<strong>池内 NAV：搬砖 +55.6%（验证期 +34.4%）vs d4 +51.3%（验证期 +26.5%）——池内插件场景两者同档，搬砖样本外略强。</strong></p>
</div>

<!-- ============ 八、偏离登记 ============ -->
<h2>八、偏离登记（继承 D1-D34，搬砖专项新增）</h2>
<div class="card">
<table>
<tr><th>编号</th><th>偏离项</th><th>说明</th><th>影响</th><th>处置</th></tr>
<tr><td><strong>D35</strong></td><td><strong>SMA 注释与实现不一致</strong></td><td><strong>zge_compute 注释称 SMA(X,N,1)≈EMA α=1/N（通达信口径 α=1/N），实际 ema() 实现 α=2/(N+1)。按代码复刻（生产行为），不按注释</strong></td><td>中（VAR6A 数值与"教科书砖型"不同，但校准证明与生产一致）</td><td>登记；不修改</td></tr>
<tr><td><strong>D36</strong></td><td><strong>atr14 缺失</strong></td><td><strong>check_banzhuan 引用 ind.get('atr14', 0) 但 calc_zge_indicators 不产出该键 → 支撑区检查退化为 低≥zd×0.995（0.5×ATR=0），仅影响附带标签 in_support_zone，不影响信号筛选</strong></td><td>低（不改变信号集）</td><td>登记</td></tr>
<tr><td><strong>D37</strong></td><td><strong>score 评分体系方向失效</strong></td><td><strong>score IC −0.028（t=−3.59），越高越差；"站上MA60"加分项组 +1.41% 低于全体 +1.54%。评分仅可作门槛（score≥5 生产口径），不可作排序</strong></td><td>高（若用 score 选股会反向）</td><td>登记；建议生产端删除排序用途</td></tr>
<tr><td>D26（继承）</td><td>原回测口径差异</td><td>原 backtest_bupiao_banzhuan.py 月频采样+8%止损+50%回撤止盈+仅池内+与补票混跑——本章全A日度全信号统一 20 日持有</td><td>中（绝对值不可与原报告直接比）</td><td>继承登记</td></tr>
<tr><td>D29（继承）</td><td>日期格式混杂</td><td>bp1_features 4 只股 928 条连字符日期——prep 已统一归一，NAV 幽灵日已消除</td><td>已修复</td><td>已处置</td></tr>
</table>
</div>

<!-- ============ 九、验收 ============ -->
<h2>九、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>9.1 前视检查</h3>
<table>
<tr><th>检查项</th><th>结果</th></tr>
<tr><td>信号时点</td><td>✅ T 日收盘后可知（var6a/砖位/支撑全用 T 日及以前数据）</td></tr>
<tr><td>入场/出场</td><td>✅ T+1 开盘 → T+20 收盘（t1_20 逐票核对一致）</td></tr>
<tr><td>校准</td><td>✅ __CAL__</td></tr>
<tr><td>右截断</td><td>✅ __N_TRUNC__ 条（__PCT_TRUNC__%）剔除</td></tr>
<tr><td>训练/验证</td><td>✅ 切分 2025-09-08，验证期未参与参数选择（生产口径来自源码）</td></tr>
</table>
<h3>9.2 最终结论</h3>
<div class="warn">
<strong>验收不通过（组合层否决 + 样本外边际翻负）。</strong><br>
<strong>不通过的理由：</strong>① 信号层边际 +0.13 pct 五信号最低，<strong>验证期边际 −0.19 pct 翻负</strong>——搬砖在样本外连宇宙都跑不赢；② 组合层 NAV +7.9% vs 基准 +31.6%、cohort 止损8% +26.8% vs +56.8%，验证期 cohort −0.6% 归零；③ 日均 2,226 仓摊薄（D32 密度机制再现）。<br>
<strong>保留的真实成分：</strong>① <strong>"多空临界"机制核心成立</strong>——dist_close IC −0.068（t=−8.57）全章最强；② VAR6A 强度单调（≥60 vs 20-40：+1.62% vs −1.45%）；③ 池内 NAV 验证期 +34.4% vs 池内基准 +33.5%——<strong>池内样本外与基准持平</strong>；④ 8% 止损切左尾有效（MAE 触发 40.4%）。<br>
<strong>降级定位：</strong>不是独立 alpha，是<strong>"强势股浅回调至多空线"的入场时机规则</strong>——可作池内择时插件候选（弱于 d4 插件的 +1.24 pct 边际），生产化须先过 walk-forward。
</div>
</div>

<!-- ============ 十、修复路径 ============ -->
<h2>十、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>#</th><th>修复项</th><th>依据</th><th>预期</th></tr>
<tr><td>1</td><td><strong>池内 + 切片过滤</strong>：chg20d 5-10% + VAR6A≥60 + pre20&lt;20% + dd20&gt;−10%</td><td>6.1/6.2 分层：最优切片 +2.23%、接盘边界明确</td><td>信号层 +2.2% 档，池外负贡献清零</td></tr>
<tr><td>2</td><td><strong>砍 score 排序用途</strong>：删除"站上MA60"加分项重测（保留 score≥5 门槛）</td><td>D37：score 负 IC、MA60 加分被数据否定</td><td>评分体系简化为 砖位+支撑 两项</td></tr>
<tr><td>3</td><td><strong>top-N 限量实验</strong>：按 dist_close 升序取每日前 N（N=10/20/50）</td><td>密度 2,226/日不可操作；dist_close 是唯一强 IC</td><td>把"贴线"从定性变成定量排序</td></tr>
<tr><td>4</td><td>与补票门禁（pre60&lt;50% &amp; dd20&gt;−15%）叠加重测</td><td>补票 A 级门禁已验证有效（+45.6→+52.5）</td><td>对手方清理复用</td></tr>
<tr><td>5</td><td>walk-forward 滚动验证</td><td>D37 选择偏差（切片来自样本内分层）</td><td>连续两段样本外为正才谈实盘</td></tr>
</table>
<p class="muted"><strong>纪律红线不变：</strong>修复验证完成前不实盘；池外信号一律不做（池外 NAV −3.8% 实证）。</p>
</div>

<div class="muted" style="margin-top:26px;text-align:center">
本报告由 A股量化研报复现专家团 Workflow B 生成 ｜ 数据源：kline_cache_all.db / zge_compute.py / stock_pool.json ｜ 口径版本 v2（2026-10-20）
</div>
</div>
</body>
</html>"""

cal_str = "%d/%d = 100.00%% 全维度一致" % (V["n_ok"], V["n_cmp"]) if V["n_cmp"] and V["n_ok"] == V["n_cmp"] else "%d/%d 不一致 %d 条" % (V["n_ok"], V["n_cmp"], len(V.get("mismatch", [])))
n_train = g["训练期"]["n"]; n_validp = g["验证期"]["n"]
n_valid = g["全体"]["n"]; n_trunc = trun["n_total"] - trun["n_truncated"]*0  # placeholder
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
html = inject(html, "搬砖(多空临界)")

out = os.path.join(HERE, "搬砖因子深度评估报告_重构版_20261020.html")
open(out, "w", encoding="utf-8").write(html)
print("saved -> %s len: %d chars" % (out, len(html)))
