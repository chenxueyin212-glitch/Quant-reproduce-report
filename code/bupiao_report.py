# -*- coding: utf-8 -*-
"""补票（单针下20）因子重构评估报告 + d4伏击实验（A级修复验证）— standalone HTML"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "bupiao_results.json"), encoding="utf-8"))
V = json.load(open(os.path.join(HERE, "bupiao_verify.json"), encoding="utf-8"))
D4 = json.load(open(os.path.join(HERE, "d4_ambush_results.json"), encoding="utf-8"))
G = json.load(open(os.path.join(HERE, "bupiao_gate_results.json"), encoding="utf-8"))

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

g = B["groups"]

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
    row("score = 6", g["score=6"]),
    row("score = 7", g["score=7"]),
    row("score ≥ 8", g["score>=8"]),
    row("标准型（dl≥85, ds<30）", g["标准型(dl>=85,ds<30)"]),
    row("激进型（dl≥80, ds<20）", g["激进型(dl>=80,ds<20)"]),
])

rows_decile = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]))
    for k, v in B["deciles"].items())

ic = B["ic"]
ic_rows = ""
for k, lbl in [("score", "补票 score（连续）"), ("dl", "单针下20_长 dl"), ("ds", "单针下20_短 ds")]:
    v = ic.get(k)
    if not v: continue
    tag = "正" if v["mean"] > 0 else "负"
    sig = "显著" if abs(v.get("t") or 0) > 2 else "不显著"
    ic_rows += ("<tr><td>%s</td><td class='%s'>%+.4f</td><td>%+.3f</td><td>%.1f%%</td><td class='%s'>%+.2f</td>"
                "<td><span class='tag %s'>%s且%s</span></td></tr>") % (
        lbl, cls(v["mean"]), v["mean"], v["icir"], v["pos_ratio"], cls(v["t"]), v["t"],
        "tag-green" if v["mean"] > 0 else "tag-red", tag, sig)

# 组合表
port_rows = ""
for k in ["不止损", "止损8%", "止损12%", "止损15%"]:
    v = B["port"][k]
    a = v["all"]; tr = v["train"]; va = v["valid"]
    if not a: continue
    port_rows += ("<tr><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td><td>%s%%</td>"
                  "<td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]), f1(a["win_m"]),
        cls(tr["total"]), f1(tr["total"]), cls(va["total"]), f1(va["total"]))
bp = B["bench_port"]["等权全A"]; bpp = B["bench_port"]["等权池内"]
port_rows += ("<tr style='background:#fdecea'><td><strong>基准·等权全A</strong></td><td class='up'><strong>%s%%</strong></td>"
              "<td><strong>%s</strong></td><td><strong>%s%%</strong></td><td><strong>%s%%</strong></td>"
              "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td></tr>") % (
    f1(bp["all"]["total"]), f2(bp["all"]["sharpe"]), f1(bp["all"]["maxdd"]), f1(bp["all"]["win_m"]),
    f1(bp["train"]["total"]), f1(bp["valid"]["total"]))

pool_rows = ""
for k in ["不止损", "止损8%"]:
    v = B["pool_port"][k]
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
for k, v in B["ofat"].items():
    if not v: continue
    bold = ' style="background:#eef4ff;font-weight:700"' if k.startswith("基准") else ""
    ofat_rows += "<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        bold, k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))

regime_rows = ""
regime_groups = []
for k in ["pre20>0(近期强势)", "pre20<=0(近期弱势)", "vol20<3%(低波)", "vol20>=3%(高波)"]:
    v = B["regime"].get(k)
    if not v: continue
    regime_rows += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), f1(v["win"]))
    regime_groups.append((k.split("(")[0], [("补票", "#e74c3c", v["mean"])]))

def micro_rows(d, name):
    out = "<tr style='background:#f8fafc'><td colspan='5'><strong>%s</strong></td></tr>" % name
    for lb, e in sorted(d.items(), key=lambda x: -x[1]["mean"] if x[1] else 0):
        if not e: continue
        out += "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td></tr>" % (
            lb, "{:,}".format(e["n"]), cls(e["mean"]), f2(e["mean"]), f2(e["median"]), f1(e["win"]))
    return out

cp_rows = ""
for k, kn in [("pre20", "前20日涨幅"), ("pre60", "前60日涨幅"), ("dd20", "20日回撤"), ("vol20", "20日波动率")]:
    cp_rows += micro_rows(B["counterparty"][k], kn)
q_rows = ""
for k, kn in [("ds(短线超卖度)", "ds 短线超卖度"), ("dl(长线趋势度)", "dl 长线趋势度"), ("chg20d", "近20日涨幅")]:
    q_rows += micro_rows(B["quality"][k], kn)

pt = B["path"]
path_rows = ("<tr><td>MAE 中位 / 均值</td><td class='down'>%s%% / %s%%</td><td>%s%%</td></tr>"
             "<tr><td>MFE 中位 / 均值</td><td>%s%% / %s%%</td><td>%s%%</td></tr>"
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

# d4 实验表
d4_rows = ""
for k in ["amb_loose(d4>=10,plat<25,未突破)", "amb_mid(d4>=12,plat<25,未突破)",
          "amb_tight(d4>=15,plat<25,未突破)", "d4_only(d4>=12)", "bp1_sig(突破后追入,对照)"]:
    v = D4.get(k)
    if not v: continue
    d4_rows += ("<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td>"
                "<td>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        k.split("(")[0], "{:,}".format(v["全部"]["n"]), cls(v["全部"]["mean"]), f2(v["全部"]["mean"]),
        cls(v["训练期"]["mean"]), f2(v["训练期"]["mean"]), f1(v["全部"]["win"]),
        cls(v["池内"]["mean"]), f2(v["池内"]["mean"]), cls(v["验证期"]["mean"]), f2(v["验证期"]["mean"]))
bl = D4["baseline"]
d4_rows += ("<tr style='background:#fdecea'><td><strong>全宇宙基准(同持有期)</strong></td><td><strong>%s</strong></td>"
            "<td class='up'><strong>%s%%</strong></td><td class='up'><strong>%s%%</strong></td>"
            "<td><strong>%s%%</strong></td><td>—</td><td class='up'><strong>%s%%</strong></td></tr>") % (
    "{:,}".format(bl["全部"]["n"]), f2(bl["全部"]["mean"]), f2(bl["训练期"]["mean"]),
    f1(bl["全部"]["win"]), f2(bl["验证期"]["mean"]))

d4_chart = bar_chart([
    ("d4>=10伏击", [("伏击", "#27ae60", D4["amb_loose(d4>=10,plat<25,未突破)"]["全部"]["mean"])]),
    ("d4>=12伏击", [("伏击", "#27ae60", D4["amb_mid(d4>=12,plat<25,未突破)"]["全部"]["mean"])]),
    ("d4>=15伏击", [("伏击", "#27ae60", D4["amb_tight(d4>=15,plat<25,未突破)"]["全部"]["mean"])]),
    ("全宇宙基准", [("基准", "#2563eb", bl["全部"]["mean"])]),
    ("bp1追入", [("追入", "#e74c3c", D4["bp1_sig(突破后追入,对照)"]["全部"]["mean"])]),
])

monthly = B["monthly"]
mon_chart = bar_chart([
    (m[:4]+"/"+m[4:], [("补票月均", "#e74c3c" if v["mean"] >= 0 else "#27ae60", v["mean"])])
    for m, v in list(monthly.items())[-18:]
])

# ---- A级：三重硬门禁重测表（5.4 节） ----
gm = G["gate_matrix"]
gate_rows = ""
for k in ["基准(无门禁)", "①chg20d>0", "②pre60<50%", "③dd20>-15%",
          "①+②", "①+③", "②+③", "①+②+③(三重门禁)"]:
    b = gm.get(k)
    if not b or not b["全体"]: continue
    a = b["全体"]; tr = b["训练期"]; va = b["验证期"]; pv = b["池内-验证期"]
    bold = ' style="background:#eef4ff;font-weight:700"' if "三重" in k else ""
    gate_rows += ("<tr%s><td>%s</td><td>%.1f%%</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td>"
                  "<td>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        bold, k, b["keep_pct"], "{:,}".format(a["n"]), cls(a["mean"]), f2(a["mean"]),
        cls(a["median"]), f2(a["median"]), f1(a["win"]),
        cls(tr["mean"]), f2(tr["mean"]), cls(va["mean"]), f2(va["mean"]),
        cls(pv["mean"]), f2(pv["mean"]))

gate_tight_rows = ""
for k in ["三重门禁(基准)", "+dl>=90", "+ds<15", "+score>=8", "+dl>=90 & ds<15", "仅dl>=90(无门禁对照)"]:
    v = G["tight"].get(k)
    if not v: continue
    bold = ' style="background:#eef4ff;font-weight:700"' if "dl>=90 &" in k else ""
    gate_tight_rows += ("<tr%s><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>") % (
        bold, k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]),
        cls(v["median"]), f2(v["median"]), f1(v["win"]))

gate_port_rows = ""
for label in ["基准(无门禁)", "②+③(pre60+dd20)", "①+②+③(三重门禁)"]:
    pv = G["port"][label]
    a = pv["止损8%"]["all"]; va = pv["止损8%"]["valid"]
    p8 = pv["池内止损8%"]["all"]; p8v = pv["池内止损8%"]["valid"]
    bold = ' style="background:#eef4ff;font-weight:700"' if "三重" in label else ""
    gate_port_rows += ("<tr%s><td>%s</td><td class='%s'>%s%%</td><td>%s</td><td>%s%%</td>"
                       "<td class='%s'>%s%%</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td></tr>") % (
        bold, label, cls(a["total"]), f1(a["total"]), f2(a["sharpe"]), f1(a["maxdd"]),
        cls(va["total"]), f1(va["total"]),
        cls(p8["total"]), f1(p8["total"]), cls(p8v["total"]), f1(p8v["total"]))

gate_chart = bar_chart([
    ("无门禁", [("基准", "#9ca3af", gm["基准(无门禁)"]["全体"]["mean"])]),
    ("①仅chg20d>0", [("单开①", "#9ca3af", gm["①chg20d>0"]["全体"]["mean"])]),
    ("②仅pre60<50%", [("单开②", "#2563eb", gm["②pre60<50%"]["全体"]["mean"])]),
    ("③仅dd20>-15%", [("单开③", "#9ca3af", gm["③dd20>-15%"]["全体"]["mean"])]),
    ("三重门禁", [("①+②+③", "#e74c3c", gm["①+②+③(三重门禁)"]["全体"]["mean"])]),
    ("三重+dl≥90", [("加严", "#27ae60", G["tight"]["+dl>=90"]["mean"])]),
])

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>补票（单针下20）因子深度评估报告 · 重构版（2026-10-20）</title>
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

<h1>补票（单针下20）因子深度评估报告 <span class="tag tag-amber">重构版 · 第四章</span></h1>
<div class="sub">生成日期：2026-10-20 ｜ 方法：A股量化研报复现专家团 Workflow B ｜ 数据：kline_cache_all.db（前复权日线，2023-06-01 ~ 2026-08-21，全A 4999 只）｜ 验收：role-07 ｜ 配套：B1 / B2 / bp1 章 + 附录 d4 伏击实验</div>

<div class="key">
<strong>验收结论先行：</strong>补票（单针下20）是<strong>四个信号中第一个"有条件可用"的</strong>——校准 600/600 = 100% 后，信号层均值 <span class="up">+1.12%</span>/20日（训练 +1.25% / 验证 +0.65%，验证期不翻负），月度 cohort 止损 8% 口径累计 <span class="up">+45.6%</span>（夏普 0.82、最大回撤 −13.4%）vs 等权全A基准 <span class="up">+56.8%</span>——<strong>绝对收益为正、风控形态健康，但仍跑输基准 11.2 pct</strong>。池内 +89.2% vs 池内基准 +123.3%。<strong>A级三重硬门禁（5.4-5.6 节）后提升至 <span class="up">+52.5%</span></strong>（夏普 0.91），与基准差距收窄到 4.3 pct。横向：B1 −3.8% &lt; bp1 −53.2% ≪ B2 +16.3% &lt; <strong>补票 +45.6%（门禁后 +52.5%）</strong>（cohort口径）。
</div>

<div class="key">
<strong>一句话给用户的答案：</strong>补票的"顺大势逆小势"叙事<strong>方向对了一半</strong>：ds 越超卖越好（ds<10 组 +1.53%）、dl 趋势越强越好（dl≥90 组 +1.76%）、<strong>只做顺势补票</strong>（chg20d&gt;0：+1.15% vs 逆势：−0.38%）、回避前 60 日已翻倍的（≥100% 组 −0.71%）。但两个致命问题：① 样本外衰减一半（+1.25%→+0.65%）；② 就算完美参数也只追平基准的八成。<strong>补票不是独立 alpha，是选股池 alpha + 超卖均值回归的组合体</strong>——去掉池内，池外仅 +0.71%。另附录给出 d4 伏击实验：<strong>波动率压缩"埋伏" +2.65% 确实好于突破后追入 −0.29%，A级修复方向成立</strong>。
</div>

<!-- ============ 一、定义 ============ -->
<h2>一、因子定义与参数（回源码逐行核对）</h2>
<div class="card">
<p>定义来源：<code>screener/zge_compute.py</code> → <code>check_bupiao(ind)</code>（补票战法：单针下20的短线超卖 + 长线趋势组合，"顺大势逆小势——错过 B1 的补救"，属逆势系统）。指标由 <code>calc_zge_indicators(klines)</code> 计算。信号日 T 日收盘后可知，<strong>T+1 开盘入场、T+20 收盘出场</strong>（本报告统一口径，与 B1/B2/bp1 可比）。</p>
<div class="formula">
<strong>单针下20 双指标：</strong><br>
· <strong>短线 ds</strong> = 100 × (C − LLV(L,3)) / (HHV(C,3) − LLV(L,3))　← 注意：<strong>HHV 用收盘价、LLV 用最低价</strong>（源码原样，不是经典随机指标）<br>
· <strong>长线 dl</strong> = 100 × (C − LLV(L,21)) / (HHV(C,21) − LLV(L,21))<br><br>
<strong>评分规则（check_bupiao 逐条对应）：</strong><br>
① <strong>标准型</strong>：dl ≥ 85 且 ds &lt; 30 → <strong>+6 分</strong>（长线强势 + 短线超卖）<br>
② <strong>激进型</strong>：dl ≥ 80 且 ds &lt; 20 → <strong>+5 分</strong>（更严的超卖换更低的门槛）<br>
③ 趋势确认：C &gt; MA60 → +1 分<br>
④ 近 20 日涨幅 chg20d &gt; 5% → +1 分；&lt; −5% 仅记"弱势补票"警告不加分<br>
<strong>生产信号口径（backtest_bupiao_banzhuan.py）：score ≥ 6</strong>
</div>
<table>
<tr><th>参数</th><th>值</th><th>参数</th><th>值</th></tr>
<tr><td>短线窗口</td><td>3 日（HHV(C,3)/LLV(L,3)）</td><td>长线窗口</td><td>21 日（HHV(C,21)/LLV(L,21)）</td></tr>
<tr><td>标准型</td><td>dl≥85 且 ds&lt;30（6分）</td><td>激进型</td><td>dl≥80 且 ds&lt;20（5分）</td></tr>
<tr><td>加分项</td><td>C&gt;MA60 (+1) / chg20d&gt;5% (+1)</td><td>生产门槛</td><td>score ≥ 6</td></tr>
<tr><td>信号日</td><td>T 日收盘后可知</td><td>入场 / 出场</td><td>T+1 开盘 / T+20 收盘</td></tr>
<tr><td>信号总量</td><td>16,519 条（完整前瞻，原始 18,669 去右截断 2,150）</td><td>宇宙</td><td>全A 4,999 只</td></tr>
</table>
<p class="muted">原回测（backtest_bupiao_banzhuan.py）为月频采样 + 8% 止损 + 50% 峰值回撤止盈、仅池内 848 只——本报告改用全A日度全信号 + 统一 20 日持有期口径，跨章可比（偏离 D26）。</p>
</div>

<h2>校准验证</h2>
<div class="card">
<div class="key">
<strong>校准结果：600/600 = 100.00% 全维度一致。</strong>方法：截断模式（truncate klines 至采样日）现算 vs 原引擎 <code>calc_zge_indicators</code> + <code>check_bupiao</code> 逐条比对 score 与信号标志。采样 30 只股 × 20 个月末日。首轮 94.83%（31 条 ±1 分差异），归因为 <code>chg_20d</code> off-by-one（源码 <code>closes[-1]/closes[-20]</code> 是 <strong>19 期</strong>收益，我方误作 20 期），修复后 100.00%。
</div>
</div>

<!-- ============ 二、口径 ============ -->
<h2>二、九项口径声明（与前三章一致）</h2>
<div class="card">
<table>
<tr><th>#</th><th>口径项</th><th>声明值</th></tr>
<tr><td>1</td><td>下期收益定义</td><td>T 日收盘后可知 → T+1 开盘入场 → T+20 收盘出场（持有 20 个交易日）</td></tr>
<tr><td>2</td><td>分组数</td><td>十分位 + 14 组分层（全体/训练/验证/池内外/交叉/score 三档/两类型）</td></tr>
<tr><td>3</td><td>多空方向</td><td>仅多头；对照基准为同宇宙等权持有</td></tr>
<tr><td>4</td><td>股票池过滤</td><td>剔除指数与名单外；未做 ST/涨跌停/停牌过滤（D2）</td></tr>
<tr><td>5</td><td>极值处理</td><td>不做 MAD 缩尾</td></tr>
<tr><td>6</td><td>中性化</td><td>不做行业/市值中性化（D12）</td></tr>
<tr><td>7</td><td>基准</td><td>等权全A（4999 只，主基准）/ 等权池内（848 只，含后视偏差）</td></tr>
<tr><td>8</td><td>无风险利率</td><td>0（月度夏普 = 均值/标准差×√12）</td></tr>
<tr><td>9</td><td>费率</td><td>主口径不计成本；费率 ≈ 单次往返 0.30%</td></tr>
</table>
<p class="muted">训练/验证切分：信号日 &lt; 2025-09-08 训练期（12,971 条）/ 验证期（3,548 条）。右截断 2,150 条（11.5%）剔除——补票月频分布使年底信号持有期易超数据截止，比例高于 bp1（1.2%），已登记 D27。</p>
</div>

<!-- ============ 三、信号层 ============ -->
<h2>三、信号层有效性检验</h2>
<div class="card">
<h3>3.1 分组统计（T+1开盘 → T+20收盘，无止损，%）</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>P10</th><th>P90</th></tr>
__ROWS_GROUPS__
</table>
<div class="note"><strong>读法：</strong>中位 −0.70% 但均值 +1.12%——仍是彩票型结构（靠右尾），但比前三章温和得多（bp1 中位 −3.33%）。<strong>验证期均值 +0.65% 是四章中唯一不翻负的样本外成绩</strong>（B2 验证期 −12.4% NAV、bp1 −14.9%）。score 分层单调（6→+0.84% / 7→+1.11% / ≥8→+1.32%）——评分体系有判别力。池内 +2.52% vs 池外 +0.71%，老规律：alpha 一半在选股池。</div>

<h3>3.2 日度截面 RankIC</h3>
<table>
<tr><th>因子</th><th>IC 均值</th><th>ICIR</th><th>IC&gt;0 占比</th><th>t 值</th><th>判定</th></tr>
__IC_ROWS__
</table>
<div class="note"><strong>ds IC = +0.038（t = 2.26）方向正确</strong>：在信号组内 ds 越低（越超卖）未来越好——与"抄短线超卖"的叙事一致。score 总分 IC 不显著（+0.006, t=0.37），因为 score 的加分项（&gt;MA60、涨5%）是趋势确认而非超卖度——<strong>真正有预测力的成分是 ds，不是加分项</strong>。</div>

<h3>3.3 收益分布（n=16,519，十分位）</h3>
<table>
<tr><th>十分位</th><th>n</th><th>组均值</th></tr>
__ROWS_DECILE__
</table>

<h3>3.4 朴素对照（同持有期）</h3>
<table>
<tr><th>组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__NAIVE_ROWS__
</table>
<div class="warn"><strong>与基准的差距必须直说：</strong>补票 +1.12% vs 全宇宙等权 +1.41%（同 20 日持有期）——<strong>信号本身还差 0.29 pct</strong>。补票的价值不在信号层 alpha（它是负的），在组合层的"风控形态"（见下节）。</div>
</div>

<!-- ============ 四、组合层 ============ -->
<h2>四、组合层绩效（月度 cohort 等权口径）</h2>
<div class="card">
<h3>4.1 止损档对照（全体）</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>月胜率%</th><th>训练期%</th><th>验证期%</th></tr>
__PORT_ROWS__
</table>
<div class="key"><strong>补票是四章中唯一"组合形态健康"的信号：</strong>止损 8% 口径 +45.6%（夏普 0.82、最大回撤仅 −13.4%、月胜率 <strong>__WINM__%</strong>）——对照 bp1 同口径 −42.7%（夏普 −1.62）。止损 8% 显著优于不止损（+17.7%），因为补票 MAE 中位 −6.18%，8% 止损触发 39.9%，<strong>切掉的是左尾而不过度扫损</strong>——与 bp1（触发 61.4%、止损是减亏不是保盈）本质不同。<strong>仍跑输基准 +56.8% 达 11.2 pct，但验证期 +3.3% vs 基准 +3.0%——样本外不再翻负（D29 修复后口径）。</strong></div>

<h3>4.2 池内组合</h3>
<table>
<tr><th>止损档</th><th>累计%</th><th>夏普</th><th>最大回撤%</th><th>训练期%</th><th>验证期%</th></tr>
__POOL_ROWS__
</table>

<h3>4.3 止损与波动率失配（MAE 分布，n=16,519）</h3>
<table>
<tr><th>阈值</th><th>MAE ≤ 阈值占比</th></tr>
__MAE_ROWS__
</table>
<div class="note">MAE 中位 −6.18%（bp1 是 −10.57%、全A基准 −6.74%）——<strong>补票的入场点回撤形态与全市场接近</strong>，不存在 bp1 那种"买在加速点"的结构性失配。这是止损 8% 能起作用的前提。</div>

<h3>4.4 月度收益（近 18 个月）</h3>
__MON_CHART__
</div>

<!-- ============ 五、稳健性 ============ -->
<h2>五、稳健性检验</h2>
<div class="card">
<h3>5.1 参数敏感性（信号层门槛）</h3>
<table>
<tr><th>变体</th><th>n</th><th>均值</th><th>胜率</th></tr>
__OFAT_ROWS__
</table>
<div class="note"><strong>最有效的三个加严方向：</strong>dl≥90（+1.76%）、ds&lt;15（+1.34%）、score≥8（+1.32%）——都在训练验证混合样本上改善。<strong>致命的是 chg20d&lt;0（逆势补票）−0.38%</strong>：生产定义允许"弱势补票"（近20日跌&gt;5%仅警告），但数据说逆势补票是负贡献——<strong>应加硬门禁 chg20d&gt;0</strong>。</div>

<h3>5.2 状态条件</h3>
<table>
<tr><th>状态</th><th>n</th><th>均值</th><th>胜率</th></tr>
__REGIME_ROWS__
</table>
__REGIME_CHART__
<div class="note">低波（+1.47%）优于高波（+0.69%）、近期强势（+1.17%）优于弱势（−0.00%）——补票吃的是"<strong>强势股的短暂超卖修复</strong>"，不是弱势股反转。dd20 −10~−5% 组（+1.72%）最好：适度回调后修复，深回撤（&lt;−15%）反而不行。</div>

<h3>5.3 数据质量与截断</h3>
<p>右截断 2,150/18,669（11.5%）——月频触发使年底信号持有期易超截止。月度信号量分布不均（见 4.4），2026 年信号量萎缩与验证期衰减同步。</p>

<h3>5.4 三重硬门禁重测（A级修复验证，2026-10-20 补测）</h3>
<p>按第十章 A 级修复路径落地重测：① chg20d&gt;0（剔逆势补票）+ ② pre60&lt;50%（剔接盘股）+ ③ dd20&gt;−15%（剔价值陷阱）。信号层逐门单开/双开/三开边际如下：</p>
<table>
<tr><th>门禁组合</th><th>保留率</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th><th>训练期</th><th>验证期</th><th>池内-验证</th></tr>
__GATE_ROWS__
</table>
__GATE_CHART__
<div class="key"><strong>门禁矩阵结论：主力是 ② pre60&lt;50%，其余两门是配角。</strong>单开②：+1.43%（保留 80.8%）；单开①：+1.15%（只剔 2.1%——score≥6 的加分规则天然偏向 chg20d&gt;0，逆势补票本来就少）；单开③：+1.15%。<strong>三重门禁：保留 78.3%（12,933 条）、均值 +1.49%（提升 +0.37 pct）、验证期 +0.82%、池内-验证期 <span class="up">+5.03%</span></strong>。被剔除的 3,586 条均值 −0.19%——剔的是薄负贡献层，不伤主样本。①+②+③ 与 ②+③ 几乎无差（+1.49% vs +1.44%）——<strong>实际生产门禁 pre60&lt;50% + dd20&gt;−15% 两门即可，chg20d&gt;0 可作廉价保险</strong>。</div>

<h3>5.5 门禁下的参数加严</h3>
<table>
<tr><th>变体（三重门禁下）</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__GATE_TIGHT_ROWS__
</table>
<div class="note"><strong>最优可操作切片：三重门禁 + dl≥90 → <span class="up">+2.54%</span>（n=1,080，胜率 52.8%）</strong>——比无门禁的 dl≥90（+1.76%）再高 0.78 pct，门禁与 dl 有正交性。dl≥90 且 ds&lt;15 双加严 +4.80%（n=211）方向极好但样本薄，只作观察。<strong>注意 +score≥8 在门禁后反而不如基准（+1.41% vs +1.49%）</strong>——score 的加分项（&gt;MA60 / 涨5%）在门禁过滤后失去增量，评分体系的核心仍是标准型/激进型的底分。</div>

<h3>5.6 门禁对组合层的影响（月度 cohort，止损 8%）</h3>
<table>
<tr><th>口径</th><th>全A累计</th><th>夏普</th><th>最大回撤</th><th>全A验证期</th><th>池内累计</th><th>池内验证期</th></tr>
__GATE_PORT_ROWS__
</table>
<div class="key"><strong>组合层验证通过：</strong>三重门禁 + 止损8% 全A cohort <span class="up">+52.5%</span>（夏普 0.91、回撤 −12.6%），比无门禁 +45.6% 提升 6.9 pct，<strong>与基准（+56.8%）的差距从 11.2 pct 收窄到 4.3 pct</strong>；验证期 +4.3%（无门禁 +3.3%）。池内口径 +86.7%（验证期 +17.5%）与无门禁 +89.2% 基本持平（池内样本被门禁剔得少，pre60 接盘股大多在池外）——<strong>门禁的增量主要来自池外垃圾层的清除</strong>。</div>
</div>

<!-- ============ 六、微观 ============ -->
<h2>六、微观经济假设检验（"顺大势逆小势"到底在买什么）</h2>
<div class="card">
<h3>6.1 质量分层：ds 超卖度与 dl 趋势度</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__Q_ROWS__
</table>
<div class="key"><strong>H1（超卖修复）验证通过一半：</strong>ds&lt;10 组 +1.53% vs ds 20-30 组 +0.97%——超卖越深越好，单调成立。dl≥90 组 +1.76% vs dl 80-85 组 +1.10%——长线趋势越强越好。<strong>"强趋势股的深超卖"是最优象限。</strong>但 chg20d 分层否定了一半机制：0-5% 组仅 +0.48%，≥10% 组 +1.23%——近 20 日没涨的补票不好，<strong>修复依赖的是"已确认的上升通道"</strong>，不是"跌不动了"。</div>

<h3>6.2 对手方分层</h3>
<table>
<tr><th>分组</th><th>n</th><th>均值</th><th>中位</th><th>胜率</th></tr>
__CP_ROWS__
</table>
<div class="warn"><strong>接盘边界清晰：</strong>pre60 ≥100%（60日翻倍）组 <span class="down">−0.71%</span>——与 B2 的 pre60 硬门禁结论完全一致（B2：&gt;20-50% 组 −0.85%）。dd20 &lt;−15% 组 −0.18%——深回撤的"超卖"是价值陷阱不是机会。<strong>可操作硬门禁：pre60 &lt; 50% 且 dd20 &gt; −15%。</strong></div>

<h3>6.3 持仓路径</h3>
<table>
<tr><th>指标</th><th>中位</th><th>均值</th></tr>
__PATH_ROWS__
</table>
<div class="note">MAE 中位 −6.18% / MFE 中位 +8.42% / 前 5 日最低 −1.51% / 后 15 日最高 +5.41%——<strong>持仓体验四章最好</strong>：低于成本天数中位 9/20 天（bp1 是 15/20），前 5 日回撤浅。补票买入的超卖修复通常很快启动，不需要"熬"。</div>
</div>

<!-- ============ 七、d4 伏击实验 ============ -->
<h2>七、附录：d4 波动率压缩伏击实验（bp1 A级修复验证）</h2>
<div class="card">
<p>背景：bp1 报告第六章发现 d4 波动率压缩是 N 类评分中唯一有正判别力的维度（秩相关 +0.141，&gt;12 分组 +5.56%），且"奖励的是突破前的静默"。A级修复假设：<strong>不追突破，在波动率压缩态埋伏</strong>。信号 = d4 ≥ 门槛 且 平台振幅 &lt;25% 且 <strong>未触发 bp1 技术五条件</strong>（未突破）。</p>
<h3>7.1 伏击 vs 追入（T+1开盘 → T+20收盘，%）</h3>
<table>
<tr><th>信号</th><th>n</th><th>均值</th><th>训练期</th><th>胜率</th><th>池内</th><th>验证期</th></tr>
__D4_ROWS__
</table>
__D4_CHART__
<div class="key"><strong>A级修复方向成立，且单调：</strong>d4 门槛越高越好（≥10：+1.74% → ≥12：+2.06% → ≥15：<span class="up">+2.65%</span>），全部跑赢全宇宙基准 +1.41%。对照 bp1 突破后追入 <span class="down">−0.29%</span>——<strong>同一套 d4 信息，"埋伏"与"追入"差 2.94 pct/20日</strong>。池内伏击 +3.28%。验证期 +1.09% vs 基准验证 +0.54%——样本外仍有 +0.55 pct alpha（衰减但不消失）。d4≥15 且池内的组合（+3.28%）是全项目迄今最好的信号切片。</div>
<div class="note"><strong>与 bp1 教训的统一解释：</strong>bp1 的技术五条件要求"放量突破"，而 bp1 微观检验已证明突破越猛/量越大未来越差——因为突破那一刻消耗了波动率压缩积蓄的全部弹性。伏击挣的是"压缩→扩张"的转换收益，追入挣的是"扩张→继续扩张"的动量——后者在 A 股 2023-2026 是负的。</div>
</div>

<!-- ============ 八、偏离登记 ============ -->
<h2>八、偏离登记（继承 D1-D24，补票专项新增）</h2>
<div class="card">
<table>
<tr><th>#</th><th>偏离项</th><th>说明</th><th>影响</th><th>等级</th></tr>
<tr><td>D1/D2/D12</td><td>无中性化 / 无 ST 过滤 / 幸存者偏差</td><td>同前三章</td><td>高</td><td>中</td></tr>
<tr><td>D25</td><td>chg_20d 口径</td><td>源码 <code>closes[-1]/closes[-20]</code> 为 19 期收益（命名 20d 不准确），本报告原样复刻</td><td>低</td><td>低</td></tr>
<tr><td><strong>D26</strong></td><td><strong>原回测口径差异</strong></td><td><strong>原 backtest_bupiao_banzhuan.py 为月频采样 + 8%止损 + 50%回撤止盈 + 仅池内；本报告为日度全信号 + 统一 20 日持有 + 全A。绝对数值不可与旧回测报告直接比（旧: 补票 14.6%胜率/+12.24%期望为信号级止损口径）</strong></td><td>中</td><td>中</td></tr>
<tr><td><strong>D27</strong></td><td><strong>右截断 11.5%</strong></td><td>2,150/18,669 条剔除（bp1 仅 1.2%）——补票月频分布使年底信号持有期易超数据截止，验证期末段样本偏少</td><td>中</td><td>低</td></tr>
<tr><td><strong>D28</strong></td><td><strong>volume 假数据</strong></td><td>校准验证时 volume 传 1000.0 常数——check_bupiao 不用 volume，不影响校准；但若未来扩展到量能相关条件需回补真实量</td><td>无</td><td>低</td></tr>
<tr><td><strong>D29</strong></td><td><strong>日期格式混杂（已修复，数字已重算）</strong></td><td><strong>bp1_features.pkl 混入 4 只股票 928 条 YYYY-MM-DD 连字符日期，r["date"][:6] 分桶产生 2025-0/2025-1/2026-0 幽灵月并插错复利顺序——旧版 cohort 数字（止损8% +56.5%、基准 +70.3%）被污染。日期归一为 YYYYMMDD 后重算：止损8% +45.6%、基准 +56.8%、验证期 +3.3%（旧 −1.0%）。本版所有数字均为修复后口径</strong></td><td><strong>高（已修）</strong></td><td>高</td></tr>
<tr><td><strong>D30</strong></td><td><strong>A级门禁重测为事后追加</strong></td><td><strong>5.4-5.6 为 2026-10-20 补测（bupiao_gate.py），门禁规则来自本报告 5.1/6.2 样本内观察再回测——存在弱选择偏差，生产前需 walk-forward 确认</strong></td><td>中</td><td>中</td></tr>
</table>
</div>

<!-- ============ 九、验收结论 ============ -->
<h2>九、验收结论（role-07 研报复现验收工程师）</h2>
<div class="card">
<h3>9.1 前视检查</h3>
<table>
<tr><th>#</th><th>检查项</th><th>结果</th></tr>
<tr><td>1</td><td>信号 T 日收盘后可知</td><td><span class="tag tag-green">通过</span> — ds/dl/MA60/chg20d 全部 ≤T 日</td></tr>
<tr><td>2</td><td>T+1 开盘入场</td><td><span class="tag tag-green">通过</span></td></tr>
<tr><td>3</td><td>指标窗口</td><td><span class="tag tag-green">通过（修复后）</span> — chg_20d off-by-one 已修（19期口径）</td></tr>
<tr><td>4</td><td>金标准比对</td><td><span class="tag tag-green">600/600 = 100.00%</span></td></tr>
<tr><td>5</td><td>右截断</td><td><span class="tag tag-amber">2,150 条剔除（11.5%）</span> — D27</td></tr>
</table>

<h3>9.2 最终结论</h3>
<div class="key">
<strong>复现过程：通过</strong>（校准 100%，唯一 off-by-one 已修复）。<br><br>
<strong>因子有效性：有条件通过——四章中第一个。</strong><br>
· 信号层：+1.12%/20日，验证期 +0.65%（<strong>唯一不翻负的样本外</strong>），ds IC +0.038（t=2.26）<br>
· 组合层：止损 8% 口径 +45.6%（夏普 0.82、回撤 −13.4%）vs 基准 +56.8%——<strong>绝对收益正、风控健康、但跑输基准 11.2 pct</strong>；<strong>A级三重门禁后 +52.5%（夏普 0.91），差距收窄至 4.3 pct</strong><br>
· 机制：强趋势股（dl≥90）+ 深超卖（ds&lt;10）+ 顺势（chg20d&gt;0）+ 无接盘（pre60&lt;50%）四条件叠加最优（门禁+dl≥90：+2.54%）<br>
· 但：池外仅 +0.71%（alpha 一半在选股池），信号层验证期 +0.65% 较训练 +1.25% 衰减约一半；<strong>组合层验证期 +3.3% 不翻负</strong>（D29 修复后口径）<br><br>
<strong>与前三章横向（月度 cohort 口径）：</strong>bp1 <span class="down">−53.2%</span> &lt; B1 −3.8% &lt; B2 +16.3% &lt; <strong>补票 +45.6%（门禁后 +52.5%）</strong> &lt; 基准 +56.8%。<br>
<strong>d4 伏击（附录）：</strong>埋伏 +2.65% vs 追入 −0.29%——bp1 的 A 级修复方向成立，d4≥15 池内 +3.28% 是全项目最好切片。
</div>
</div>

<!-- ============ 十、修复路径 ============ -->
<h2>十、基于数据的修复路径（按优先级）</h2>
<div class="card">
<table>
<tr><th>优先级</th><th>方向</th><th>依据</th></tr>
<tr><td><strong>A ✅已完成</strong></td><td><strong>补票三重硬门禁</strong>：chg20d&gt;0 + pre60&lt;50% + dd20&gt;−15%（见 5.4-5.6）。<strong>实测：保留 78.3%、均值 +1.12%→+1.49%、cohort 止损8% +45.6%→+52.5%（夏普 0.91）</strong>——实际增量主力是 pre60&lt;50%（单开 +1.43%），生产门禁 pre60&lt;50% + dd20&gt;−15% 两门即可</td><td>5.4</td></tr>
<tr><td><strong>A</strong></td><td><strong>参数加严</strong>：三重门禁 + dl≥90 → <strong>+2.54%</strong>（n=1,080）为最优可操作切片；+score≥8 门禁后无增量（+1.41%），放弃</td><td>5.5</td></tr>
<tr><td><strong>S ✅已交付</strong></td><td><strong>d4 伏击因子产品化</strong>：完整五项检验已交付独立报告《d4伏击因子深度评估报告_20261020》。<strong>结论：产品化不通过（NAV +6.2% vs 基准 +31.6%），降级为池内择时插件（池内 NAV +51.3%、验证期 +26.5%）</strong>——信号层 +2.65% vs 宇宙 +1.41% 成立，组合层因信号密度过高（日均 4,118 仓）摊薄失败</td><td>第七章 / 独立报告</td></tr>
<tr><td>B</td><td>止盈改造：MFE 中位 +8.42% 但均值 +13.4%——右尾长，50% 峰值回撤止盈（原回测用）可能优于固定 20 日到期，可测 10/20/30 日 × 固定/移动矩阵</td><td>6.3</td></tr>
<tr><td>C</td><td>池内限定：池内 +2.52%（止损8% cohort +89.2%）——若私域池可用，补票是池内择时器而非全A策略</td><td>3.1 / 4.2</td></tr>
<tr><td>D</td><td>验证期衰减归因：+1.25%→+0.65%，与 2026 年信号量萎缩同步——按月拆解确认是市场状态问题还是参数失效</td><td>4.4 / 5.3</td></tr>
</table>
<div class="key">
<strong>四章总结论（更新，含 A/S 级修复结果）：</strong>没有任何一个信号跑赢"等权持有"基准，但质量阶梯清晰——bp1 追突破是<strong>结构性反指</strong>（−53.2%），B1/B2 是<strong>无增量</strong>（−3.8%/+16.3%），补票是<strong>正期望但不足额</strong>（+45.6%，门禁后 +52.5% vs 基准 +56.8%）。所有信号的池内版本都大幅好于池外——<strong>选股池 alpha 仍是唯一可靠的底仓，信号只配做池内择时</strong>。A 级门禁已落地（5.4-5.6），S 级 d4 伏击已交付独立报告（结论：产品化不通过、池内可用）。下一优先级：① 池内伏击 top-N 限量实验 ② 剩余信号（bp2/搬砖）同标准扫描 ③ 门禁版补票的小仓位实盘对照。
</div>
</div>

<div class="muted" style="margin-top:30px;text-align:center">
补票（单针下20）因子深度评估报告 · 重构版 ｜ 2026-10-20 ｜ A股量化研报复现专家团 Workflow B ｜ 配套：B1 / B2 / bp1 章
</div>

</div>
</body>
</html>"""

naive_rows = "".join(
    "<tr><td>%s</td><td>%s</td><td class='%s'>%s%%</td><td class='%s'>%s%%</td><td>%s%%</td></tr>" % (
        k, "{:,}".format(v["n"]), cls(v["mean"]), f2(v["mean"]), cls(v["median"]), f2(v["median"]), f1(v["win"]))
    for k, v in B["naive"].items() if v)

html = html.replace("__ROWS_GROUPS__", rows_groups)
html = html.replace("__ROWS_DECILE__", rows_decile)
html = html.replace("__IC_ROWS__", ic_rows)
html = html.replace("__PORT_ROWS__", port_rows)
html = html.replace("__POOL_ROWS__", pool_rows)
html = html.replace("__NAIVE_ROWS__", naive_rows)
html = html.replace("__OFAT_ROWS__", ofat_rows)
html = html.replace("__REGIME_ROWS__", regime_rows)
html = html.replace("__REGIME_CHART__", bar_chart(regime_groups))
html = html.replace("__CP_ROWS__", cp_rows)
html = html.replace("__Q_ROWS__", q_rows)
html = html.replace("__PATH_ROWS__", path_rows)
html = html.replace("__MAE_ROWS__", mae_rows)
html = html.replace("__MON_CHART__", mon_chart)
html = html.replace("__D4_ROWS__", d4_rows)
html = html.replace("__D4_CHART__", d4_chart)
html = html.replace("__GATE_ROWS__", gate_rows)
html = html.replace("__GATE_TIGHT_ROWS__", gate_tight_rows)
html = html.replace("__GATE_PORT_ROWS__", gate_port_rows)
html = html.replace("__GATE_CHART__", gate_chart)
html = html.replace("__WINM__", f1(B["port"]["止损8%"]["all"]["win_m"]))

from healthcheck_chapter import inject
html = inject(html, "补票(单针下20)")

out = os.path.join(HERE, "补票因子深度评估报告_重构版_20261020.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print("saved ->", out, "len:", len(html))
