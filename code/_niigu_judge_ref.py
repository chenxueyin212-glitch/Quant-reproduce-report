# -*- coding: utf-8 -*-
"""
牛股判断模块 V1.0 -- 大牛股启动前八大特征量化评分
=================================================

基于Kimi总结的大牛股启动前8大特征，逐项量化为可每日重复执行的技术指标。
与顺势系统的A类/D类评分互补，形成N类(牛股判断)独立评分维度。
注: 顺势系统(原融合系统V2.1) + 逆势系统(原Z哥系统) 共同构成天才游资系统V1.0。

八大特征 -> 六维量化 + 两维标记:
  (1) 趋势结构(25分): 均线多头排列、第二阶段(上升趋势)判定
  (2) 量价关系(20分): 呼吸量能(底部缩量 -> 突破放量节奏)
  (3) 相对强弱(20分): RPS相对价格强度、大盘超额收益
  (4) 波动率压缩(15分): BBW/ATR历史低分位(爆发前夜特征)
  (5) 技术形态(已移除): 与Kimi买点重合，V3.3起不再参与评分（2026-06-30）
  (6) 筹码结构(10分): 流通市值范围、换手率健康度
  (7) 催化与预期差(标记): 灵魂信号 -- 题材/业绩拐点，需人工判定
  (8) 时间窗口(过滤): 大盘环境过滤条件(乖离折扣)
  [修正] 阶段新鲜度(系数0.82~1.0, 2026-07-20新增): 替代技术形态维度的角色,
         对"延伸度过高(收盘>>MA60)/趋势时长过老(MA20上穿MA60过久)"的末段票向下修正,
         纠正原N类"高分=趋势末端"偏置(N≥60回测负期望, 中等N胜率最高)

总分: 0-100分
评级:
  N >= 80: *** 大牛股候选(六维均优)
  N >= 60: **  强势股(核心维度达标)
  N >= 40: *   潜力股(部分维度突出)
  N <  40: --  不达标

集成方式:
  from niugu_judge import compute_niugu_score
  result = compute_niugu_score(klines, all_closes_map=all_closes_map)
  # result['n_score'] 总分
  # result['n_grade'] 评级
  # result['dimensions'] 各维度详情

作者: WorkBuddy Agent for 天才游资潘小铠
日期: 2026-06-14
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================
# 工具函数
# ============================================================

def _safe_get(arr, idx):
    """安全取值，支持负索引，越界或NaN返回None"""
    if arr is None:
        return None
    if idx < 0:
        idx = len(arr) + idx  # 将负索引转换为正索引
    if idx < 0 or idx >= len(arr):
        return None
    v = arr[idx]
    return None if (v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))) else float(v)


def _ma(arr, period):
    """简单移动平均，返回等长数组"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result


def _ema(arr, period):
    """指数移动平均"""
    n = len(arr)
    result = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    # 找第一个连续period个非NaN的起始位置
    start_idx = -1
    for i in range(n - period + 1):
        if not any(np.isnan(arr[i:i + period])):
            start_idx = i
            break
    if start_idx < 0:
        return result
    init_idx = start_idx + period - 1
    result[init_idx] = np.mean(arr[start_idx:start_idx + period])
    for i in range(init_idx + 1, n):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _percentile_rank(values, current):
    """
    计算current在values序列中的百分位排名(0-100)。
    返回: 有多少比例的值 <= current
    """
    if not values or current is None:
        return 50.0
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return 50.0
    return float(np.sum(arr <= current) / len(arr) * 100)


# ============================================================
# 维度1: 趋势结构 (25分)
# ============================================================

def _dim_trend_structure(closes, highs, lows):
    """
    趋势结构评分 -- 均线多头排列 + 趋势阶段判定

    评分细则:
    - 均线排列(最高15分):
      MA5 > MA10 > MA20 > MA60 完美多头: 15分
      MA5 > MA20 > MA60 三线多头: 12分
      Close > MA20 > MA60 基础多头: 8分
      Close > MA60 不破位: 4分
      Close < MA60 破位: 0分
    - MA20方向(最高5分):
      MA20斜率>0.3%/日: 5分
      MA20斜率>0: 3分
      MA20斜率<0: 0分
    - MA60方向(最高3分):
      MA60斜率>0.1%/日: 3分
      MA60斜率>0: 2分
      MA60斜率<0: 0分
    - 均线发散(最高2分):
      价格>MA5>MA10>MA20>MA60且发散: 2分

    总分上限25分
    """
    n = len(closes)
    score = 0
    details = {}

    ma5 = _ma(closes, 5)
    ma10 = _ma(closes, 10)
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)

    last_close = closes[-1]
    last_ma5 = _safe_get(ma5, -1)
    last_ma10 = _safe_get(ma10, -1)
    last_ma20 = _safe_get(ma20, -1)
    last_ma60 = _safe_get(ma60, -1)

    # === 均线排列 (最高15分) ===
    alignment_score = 0
    alignment_desc = ""

    if all(v is not None for v in [last_ma5, last_ma10, last_ma20, last_ma60]):
        if last_ma5 > last_ma10 > last_ma20 > last_ma60:
            alignment_score = 15
            alignment_desc = "完美多头排列(MA5>MA10>MA20>MA60)"
        elif last_ma5 > last_ma20 > last_ma60:
            alignment_score = 12
            alignment_desc = "三线多头(MA5>MA20>MA60)"
        elif last_close > last_ma20 > last_ma60:
            alignment_score = 8
            alignment_desc = "基础多头(C>MA20>MA60)"
        elif last_close > last_ma60:
            alignment_score = 4
            alignment_desc = "不破MA60"
        else:
            alignment_score = 0
            alignment_desc = f"破位(C={last_close:.2f}<MA60={last_ma60:.2f})"
    elif last_ma20 is not None and last_ma60 is not None:
        if last_close > last_ma20 > last_ma60:
            alignment_score = 8
            alignment_desc = "基础多头(C>MA20>MA60)"
        elif last_close > last_ma60:
            alignment_score = 4
            alignment_desc = "不破MA60"
        else:
            alignment_score = 0
            alignment_desc = "破位"
    elif last_ma60 is not None:
        if last_close > last_ma60:
            alignment_score = 4
            alignment_desc = "不破MA60"
        else:
            alignment_score = 0
            alignment_desc = "破位"
    else:
        alignment_score = 0
        alignment_desc = "MA60数据不足"

    score += alignment_score
    details['均线排列'] = {'score': alignment_score, 'desc': alignment_desc}

    # === MA20方向 (最高5分) ===
    ma20_slope_score = 0
    if n >= 25 and last_ma20 is not None:
        ma20_5ago = _safe_get(ma20, -6)
        if ma20_5ago is not None and ma20_5ago > 0:
            slope_pct = (last_ma20 / ma20_5ago - 1) / 5 * 100  # 日均斜率%
            if slope_pct > 0.3:
                ma20_slope_score = 5
                details['MA20方向'] = {'score': 5, 'desc': f'MA20斜率{slope_pct:.2f}%/日 强势向上'}
            elif slope_pct > 0:
                ma20_slope_score = 3
                details['MA20方向'] = {'score': 3, 'desc': f'MA20斜率{slope_pct:.2f}%/日 温和向上'}
            else:
                ma20_slope_score = 0
                details['MA20方向'] = {'score': 0, 'desc': f'MA20斜率{slope_pct:.2f}%/日 向下'}
        else:
            details['MA20方向'] = {'score': 0, 'desc': 'MA20参考值不足'}
    else:
        details['MA20方向'] = {'score': 0, 'desc': 'MA20数据不足'}

    score += ma20_slope_score

    # === MA60方向 (最高3分) ===
    ma60_slope_score = 0
    if n >= 65 and last_ma60 is not None:
        ma60_5ago = _safe_get(ma60, -6)
        if ma60_5ago is not None and ma60_5ago > 0:
            slope_pct = (last_ma60 / ma60_5ago - 1) / 5 * 100
            if slope_pct > 0.1:
                ma60_slope_score = 3
                details['MA60方向'] = {'score': 3, 'desc': f'MA60斜率{slope_pct:.2f}%/日 向上'}
            elif slope_pct > 0:
                ma60_slope_score = 2
                details['MA60方向'] = {'score': 2, 'desc': f'MA60斜率{slope_pct:.2f}%/日 微向上'}
            else:
                ma60_slope_score = 0
                details['MA60方向'] = {'score': 0, 'desc': f'MA60斜率{slope_pct:.2f}%/日 向下'}
        else:
            details['MA60方向'] = {'score': 0, 'desc': 'MA60参考值不足'}
    else:
        details['MA60方向'] = {'score': 0, 'desc': 'MA60数据不足(需65日)'}

    score += ma60_slope_score

    # === 均线发散 (最高2分) ===
    divergence_score = 0
    if all(v is not None for v in [last_ma5, last_ma10, last_ma20, last_ma60]):
        if last_ma5 > last_ma10 > last_ma20 > last_ma60:
            # 检查各均线间距是否在扩大(5日前对比)
            ma5_5ago = _safe_get(ma5, -6)
            ma10_5ago = _safe_get(ma10, -6)
            ma20_5ago = _safe_get(ma20, -6)
            ma60_5ago = _safe_get(ma60, -6)
            if all(v is not None for v in [ma5_5ago, ma10_5ago, ma20_5ago, ma60_5ago]):
                spread_now = (last_ma5 - last_ma60) / last_ma60
                spread_5ago = (ma5_5ago - ma60_5ago) / ma60_5ago
                if spread_now > spread_5ago + 0.01:  # 发散扩大1%以上
                    divergence_score = 2
                    details['均线发散'] = {'score': 2, 'desc': '均线发散扩大中'}
                else:
                    divergence_score = 0
                    details['均线发散'] = {'score': 0, 'desc': '均线收敛或平行'}
            else:
                details['均线发散'] = {'score': 0, 'desc': '参考数据不足'}
        else:
            details['均线发散'] = {'score': 0, 'desc': '非多头排列不判发散'}
    else:
        details['均线发散'] = {'score': 0, 'desc': '均线数据不全'}

    score += divergence_score

    # 钳位到25分
    score = min(score, 25)

    return score, details


# ============================================================
# 维度2: 量价关系 (20分)
# ============================================================

def _dim_volume_price(closes, volumes, opens, lows, highs):
    """
    量价关系评分 -- 呼吸量能(底部缩量 -> 突破放量节奏)

    评分细则:
    A. 量能呼吸节奏 (最高10分):
       近20日存在缩量期(量<0.6倍均量) + 当前放量(量比>1.5): 10分
       近20日存在缩量期 + 量比>1.2: 7分
       近20日存在缩量期但未放量: 4分
       无明显缩量->放量节奏: 0分
    B. 量价配合度 (最高5分):
       近5日量价正相关系数 > 0.5: 5分(量增价增)
       近5日量价相关系数 > 0: 3分
       量价背离(量增价跌): 0分
    C. 底部放量特征 (最高5分):
       近60日最低价附近出现量比>2.0的阳线: 5分
       近60日最低价附近出现量比>1.5的阳线: 3分
       无底部放量: 0分
    """
    n = len(closes)
    score = 0
    details = {}

    # 基础量能指标
    vol_ma5 = _ma(volumes, 5)
    vol_ma20 = _ma(volumes, 20)
    last_vol = volumes[-1]
    last_vol_ma5 = _safe_get(vol_ma5, -1)
    last_vol_ma20 = _safe_get(vol_ma20, -1)

    vol_ratio_5 = last_vol / last_vol_ma5 if (last_vol_ma5 and last_vol_ma5 > 0) else 1.0
    vol_ratio_20 = last_vol / last_vol_ma20 if (last_vol_ma20 and last_vol_ma20 > 0) else 1.0

    # === A. 量能呼吸节奏 (最高10分) ===
    breath_score = 0

    # 检测近20日是否存在缩量期
    has_shrink_period = False
    shrink_days = 0
    if n >= 20:
        for i in range(-20, 0):
            v_ma = _safe_get(vol_ma20, i)
            if v_ma and v_ma > 0 and volumes[i] < v_ma * 0.6:
                has_shrink_period = True
                shrink_days += 1

    if has_shrink_period and vol_ratio_5 > 1.5:
        breath_score = 10
        details['量能呼吸'] = {'score': 10, 'desc': f'缩量{shrink_days}日后放量突破(量比{vol_ratio_5:.1f})'}
    elif has_shrink_period and vol_ratio_5 > 1.2:
        breath_score = 7
        details['量能呼吸'] = {'score': 7, 'desc': f'缩量{shrink_days}日后温和放量(量比{vol_ratio_5:.1f})'}
    elif has_shrink_period:
        breath_score = 4
        details['量能呼吸'] = {'score': 4, 'desc': f'有缩量期{shrink_days}日但尚未放量(量比{vol_ratio_5:.1f})'}
    else:
        breath_score = 0
        details['量能呼吸'] = {'score': 0, 'desc': '无明显缩量->放量节奏'}

    score += breath_score

    # === B. 量价配合度 (最高5分) ===
    vp_score = 0
    if n >= 10:
        # 近5日量变化率和价变化率的相关系数
        recent_vol = volumes[-6:]
        recent_close = closes[-6:]
        vol_chg = np.diff(recent_vol)
        close_chg = np.diff(recent_close)

        if len(vol_chg) >= 2 and np.std(vol_chg) > 0 and np.std(close_chg) > 0:
            corr = np.corrcoef(vol_chg, close_chg)[0, 1]
            if not np.isnan(corr):
                if corr > 0.5:
                    vp_score = 5
                    details['量价配合'] = {'score': 5, 'desc': f'量价正相关{corr:.2f}(量增价增)'}
                elif corr > 0:
                    vp_score = 3
                    details['量价配合'] = {'score': 3, 'desc': f'量价弱正相关{corr:.2f}'}
                else:
                    vp_score = 0
                    details['量价配合'] = {'score': 0, 'desc': f'量价背离(相关{corr:.2f})'}
            else:
                details['量价配合'] = {'score': 0, 'desc': '数据不足'}
        else:
            details['量价配合'] = {'score': 0, 'desc': '波动不足无法计算'}
    else:
        details['量价配合'] = {'score': 0, 'desc': '数据不足(需10日)'}

    score += vp_score

    # === C. 底部放量特征 (最高5分) ===
    bottom_score = 0
    if n >= 30:
        # 找近60日(或可用数据)的最低价位置
        lookback = min(60, n)
        recent_lows = lows[-lookback:] if n >= 60 else lows
        min_low_idx = np.argmin(recent_lows)
        # 转换到全局索引
        global_min_idx = n - lookback + min_low_idx

        # 检查最低价附近3日内是否有放量阳线
        for offset in range(-1, 4):  # 最低日前1天到后3天
            check_idx = global_min_idx + offset
            if 0 <= check_idx < n:
                local_vol_ma = _safe_get(vol_ma20, check_idx)
                local_vol_ratio = volumes[check_idx] / local_vol_ma if (local_vol_ma and local_vol_ma > 0) else 1.0
                is_bullish = closes[check_idx] > opens[check_idx] if check_idx < len(opens) else False

                if is_bullish and local_vol_ratio > 2.0:
                    bottom_score = 5
                    details['底部放量'] = {'score': 5, 'desc': f'底部出现量比{local_vol_ratio:.1f}阳线'}
                    break
                elif is_bullish and local_vol_ratio > 1.5:
                    bottom_score = 3
                    details['底部放量'] = {'score': 3, 'desc': f'底部出现量比{local_vol_ratio:.1f}阳线'}
                    break

        if bottom_score == 0:
            details['底部放量'] = {'score': 0, 'desc': '底部无显著放量特征'}
    else:
        details['底部放量'] = {'score': 0, 'desc': '数据不足(需30日)'}

    score += bottom_score

    # 钳位到20分
    score = min(score, 20)

    return score, details


# ============================================================
# 维度3: 相对强弱 (20分)
# ============================================================

def _dim_relative_strength(closes, all_closes_map=None, code=None):
    """
    相对强弱评分 -- RPS相对价格强度 + 大盘超额收益

    评分细则:
    如果有全市场数据(all_closes_map):
      - RPS(250日涨幅百分位) >= 90: 20分
      - RPS >= 80: 16分
      - RPS >= 70: 12分
      - RPS >= 50: 8分
      - RPS >= 30: 4分
      - RPS < 30: 0分

    如果无全市场数据(退化为动量模式):
      - 近120日涨幅 > 30%: 16分
      - 近120日涨幅 > 15%: 12分
      - 近120日涨幅 > 5%: 8分
      - 近120日涨幅 > 0%: 4分
      - 涨幅 <= 0%: 0分

    附加(最多+4分，不超上限):
      - 近20日创60日新高: +2分
      - 近5日涨幅>3%: +2分
    """
    n = len(closes)
    score = 0
    details = {}

    # === RPS 或动量评分 ===
    if all_closes_map and len(all_closes_map) >= 10:
        # 有全市场数据 -> 计算真正的RPS
        rps_period = min(250, n)
        my_return = (closes[-1] / closes[-rps_period] - 1) * 100 if n >= rps_period else 0

        # 计算全市场所有股票同期涨幅
        all_returns = []
        for c_code, c_closes in all_closes_map.items():
            if c_closes is not None and len(c_closes) >= rps_period:
                ret = (c_closes[-1] / c_closes[-rps_period] - 1) * 100
                all_returns.append(ret)

        if all_returns:
            rps = _percentile_rank(all_returns, my_return)
            details['RPS'] = {'value': round(rps, 1), 'period': rps_period,
                              'my_return': round(my_return, 1), 'sample_size': len(all_returns)}
            if rps >= 90:
                score = 20
            elif rps >= 80:
                score = 16
            elif rps >= 70:
                score = 12
            elif rps >= 50:
                score = 8
            elif rps >= 30:
                score = 4
            else:
                score = 0
            details['RPS']['score'] = score
        else:
            # 退化为动量模式
            score = _momentum_score(closes, n, details)
    else:
        # 无全市场数据 -> 动量模式
        score = _momentum_score(closes, n, details)

    # === 附加分: 创新高 ===
    if n >= 60:
        high_60 = np.max(closes[-60:])
        if closes[-1] >= high_60 * 0.98:  # 接近或创60日新高
            score += 2
            details['创新高'] = {'score': 2, 'desc': f'接近60日新高{high_60:.2f}'}

    # === 附加分: 近期强势 ===
    if n >= 5:
        chg_5d = (closes[-1] / closes[-6] - 1) * 100
        if chg_5d > 3:
            score += 2
            details['近期强势'] = {'score': 2, 'desc': f'近5日涨{chg_5d:.1f}%'}

    # 钳位到20分
    score = min(score, 20)

    return score, details


def _momentum_score(closes, n, details):
    """动量模式(无全市场数据时的退化评分)"""
    momentum_period = min(120, n)
    if n >= momentum_period:
        ret = (closes[-1] / closes[-momentum_period] - 1) * 100
    else:
        ret = 0

    if ret > 30:
        s = 16
    elif ret > 15:
        s = 12
    elif ret > 5:
        s = 8
    elif ret > 0:
        s = 4
    else:
        s = 0

    details['动量'] = {'score': s, 'desc': f'近{momentum_period}日涨幅{ret:.1f}%(动量模式)'}
    return s


# ============================================================
# 维度4: 波动率压缩 (15分)
# ============================================================

def _dim_volatility_compression(closes, highs, lows):
    """
    波动率压缩评分 -- BBW/ATR历史低分位(爆发前夜特征)

    核心逻辑: 大牛股启动前，波动率通常压缩到历史极低水平
    (如同弹簧被极度压缩，蓄势待发)

    评分细则:
    BBW百分位(近120日):
      <= 10%分位(极度压缩): 10分
      <= 20%分位: 8分
      <= 30%分位: 6分
      <= 50%分位: 4分
      > 50%分位: 2分
      > 80%分位(扩张): 0分

    ATR/Close百分位(近120日):
      同上分级，最高5分

    取BBW分(最高10分) + ATR分(最高5分) = 总分(最高15分)
    """
    n = len(closes)
    score = 0
    details = {}

    lookback = min(120, n)

    # === BBW (布林带宽度) ===
    bbw_score = 0
    if n >= 30:
        period = 20
        # 计算布林带
        sma20 = _ma(closes, period)
        # 标准差
        std20 = np.full(n, np.nan)
        for i in range(period - 1, n):
            std20[i] = np.std(closes[i - period + 1:i + 1])

        # BBW = (Upper - Lower) / Middle
        bbw = np.full(n, np.nan)
        for i in range(period - 1, n):
            if sma20[i] > 0 and not np.isnan(std20[i]):
                upper = sma20[i] + 2 * std20[i]
                lower = sma20[i] - 2 * std20[i]
                bbw[i] = (upper - lower) / sma20[i]

        # 计算百分位
        bbw_series = bbw[-lookback:]
        bbw_valid = bbw_series[~np.isnan(bbw_series)]
        current_bbw = bbw[-1] if not np.isnan(bbw[-1]) else None

        if current_bbw is not None and len(bbw_valid) >= 10:
            bbw_pct = _percentile_rank(bbw_valid.tolist(), current_bbw)
            details['BBW分位'] = {'value': round(bbw_pct, 1), 'current_bbw': round(current_bbw, 4)}

            if bbw_pct <= 10:
                bbw_score = 10
                details['BBW分位']['desc'] = '极度压缩(<=10%分位) -- 爆发前夜!'
            elif bbw_pct <= 20:
                bbw_score = 8
                details['BBW分位']['desc'] = '高度压缩(<=20%分位)'
            elif bbw_pct <= 30:
                bbw_score = 6
                details['BBW分位']['desc'] = '中度压缩(<=30%分位)'
            elif bbw_pct <= 50:
                bbw_score = 4
                details['BBW分位']['desc'] = '轻度压缩(<=50%分位)'
            elif bbw_pct <= 80:
                bbw_score = 2
                details['BBW分位']['desc'] = '波动正常'
            else:
                bbw_score = 0
                details['BBW分位']['desc'] = '波动扩张(>80%分位)'

            details['BBW分位']['score'] = bbw_score
        else:
            details['BBW分位'] = {'score': 0, 'desc': 'BBW数据不足'}
    else:
        details['BBW分位'] = {'score': 0, 'desc': '数据不足(需30日)'}

    score += bbw_score

    # === ATR/Close 百分位 ===
    atr_score = 0
    atr_period = 14
    if n >= atr_period + 10:
        # 计算ATR序列
        tr = np.full(n, np.nan)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        atr = _ma(tr, atr_period)
        atr_ratio = atr / closes  # ATR/Close 归一化

        atr_ratio_series = atr_ratio[-lookback:]
        atr_ratio_valid = atr_ratio_series[~np.isnan(atr_ratio_series)]
        current_atr_ratio = atr_ratio[-1] if not np.isnan(atr_ratio[-1]) else None

        if current_atr_ratio is not None and len(atr_ratio_valid) >= 10:
            atr_pct = _percentile_rank(atr_ratio_valid.tolist(), current_atr_ratio)
            details['ATR分位'] = {'value': round(atr_pct, 1), 'current_atr_ratio': round(current_atr_ratio, 4)}

            if atr_pct <= 10:
                atr_score = 5
                details['ATR分位']['desc'] = '极度压缩(<=10%分位)'
            elif atr_pct <= 20:
                atr_score = 4
                details['ATR分位']['desc'] = '高度压缩(<=20%分位)'
            elif atr_pct <= 30:
                atr_score = 3
                details['ATR分位']['desc'] = '中度压缩(<=30%分位)'
            elif atr_pct <= 50:
                atr_score = 2
                details['ATR分位']['desc'] = '轻度压缩(<=50%分位)'
            elif atr_pct <= 80:
                atr_score = 1
                details['ATR分位']['desc'] = '波动正常'
            else:
                atr_score = 0
                details['ATR分位']['desc'] = '波动扩张(>80%分位)'

            details['ATR分位']['score'] = atr_score
        else:
            details['ATR分位'] = {'score': 0, 'desc': 'ATR数据不足'}
    else:
        details['ATR分位'] = {'score': 0, 'desc': '数据不足(需24日)'}

    score += atr_score

    # 钳位到15分
    score = min(score, 15)

    return score, details


# ============================================================
# 维度5: 技术形态 (10分)
# ============================================================

def _dim_technical_pattern(closes, highs, lows, volumes):
    """
    技术形态评分 -- VCP波动率收缩近似 + 平台突破检测

    VCP(Volatility Contraction Pattern):
    大牛股启动前常呈现"波动幅度递减"的收缩形态，
    每次回调幅度比上一次更小，说明浮动筹码被充分消化。

    评分细则:
    A. VCP波动率收缩 (最高7分):
       将近60日分为3-4段，检测每段波动幅度是否递减:
       3次递减收缩: 7分
       2次递减收缩: 5分
       1次递减收缩: 3分
       无收缩: 0分

    B. 平台突破 (最高3分):
       近20日价格波动范围<10% + 当日接近或突破上沿:
       突破20日高点: 3分
       接近20日高点(95%): 2分
       窄幅整理但未突破: 1分
       无平台特征: 0分
    """
    n = len(closes)
    score = 0
    details = {}

    # === A. VCP波动率收缩 (最高7分) ===
    vcp_score = 0
    lookback = min(60, n)

    if lookback >= 30:
        # 将近lookback日分为3-4段
        num_segments = min(4, lookback // 10)
        if num_segments >= 2:
            segment_size = lookback // num_segments
            amplitudes = []

            for seg in range(num_segments):
                start = n - lookback + seg * segment_size
                end = start + segment_size
                seg_high = np.max(highs[start:end])
                seg_low = np.min(lows[start:end])
                if seg_low > 0:
                    amplitude = (seg_high - seg_low) / seg_low * 100
                    amplitudes.append(amplitude)

            # 检测递减收缩次数
            contractions = 0
            for i in range(1, len(amplitudes)):
                if amplitudes[i] < amplitudes[i - 1]:
                    contractions += 1

            if contractions >= 3:
                vcp_score = 7
                details['VCP收缩'] = {'score': 7, 'desc': f'3次递减收缩({[round(a, 1) for a in amplitudes]})'}
            elif contractions >= 2:
                vcp_score = 5
                details['VCP收缩'] = {'score': 5, 'desc': f'2次递减收缩({[round(a, 1) for a in amplitudes]})'}
            elif contractions >= 1:
                vcp_score = 3
                details['VCP收缩'] = {'score': 3, 'desc': f'1次递减收缩({[round(a, 1) for a in amplitudes]})'}
            else:
                vcp_score = 0
                details['VCP收缩'] = {'score': 0, 'desc': f'无收缩(波动{[round(a, 1) for a in amplitudes]})'}
        else:
            details['VCP收缩'] = {'score': 0, 'desc': '分段不足'}
    else:
        details['VCP收缩'] = {'score': 0, 'desc': '数据不足(需30日)'}

    score += vcp_score

    # === B. 平台突破 (最高3分) ===
    platform_score = 0
    if n >= 25:
        # 近20日价格范围
        recent_20_high = np.max(highs[-20:])
        recent_20_low = np.min(lows[-20:])
        range_pct = (recent_20_high - recent_20_low) / recent_20_low * 100

        last_close = closes[-1]

        if range_pct < 10:  # 窄幅整理
            # 检查是否突破
            if last_close >= recent_20_high * 0.99:
                platform_score = 3
                details['平台突破'] = {'score': 3, 'desc': f'窄幅整理{range_pct:.1f}%后突破20日高点{recent_20_high:.2f}'}
            elif last_close >= recent_20_high * 0.95:
                platform_score = 2
                details['平台突破'] = {'score': 2, 'desc': f'窄幅整理{range_pct:.1f}%接近20日高点(95%)'}
            else:
                platform_score = 1
                details['平台突破'] = {'score': 1, 'desc': f'窄幅整理{range_pct:.1f}%但未突破'}
        elif range_pct < 15:
            if last_close >= recent_20_high * 0.98:
                platform_score = 2
                details['平台突破'] = {'score': 2, 'desc': f'区间整理{range_pct:.1f}%后接近突破'}
            else:
                platform_score = 0
                details['平台突破'] = {'score': 0, 'desc': f'区间整理{range_pct:.1f}%未突破'}
        else:
            platform_score = 0
            details['平台突破'] = {'score': 0, 'desc': f'波动范围{range_pct:.1f}%过宽，非平台形态'}
    else:
        details['平台突破'] = {'score': 0, 'desc': '数据不足(需25日)'}

    score += platform_score

    # 钳位到10分
    score = min(score, 10)

    return score, details


# ============================================================
# 维度6: 筹码结构 (10分)
# ============================================================

def _dim_chip_structure(closes, volumes, extra_data=None):
    """
    筹码结构评分 -- 流通市值范围 + 换手率健康度

    评分细则:
    A. 流通市值适中性 (最高5分):
       50-500亿: 5分(最佳范围，大资金进得去、拉得动)
       20-50亿或500-1000亿: 3分
       <20亿或>1000亿: 1分
       无数据: 3分(默认中间值，避免惩罚)

    B. 换手率健康度 (最高5分):
       3%-8%: 5分(活跃适中，流动性好)
       1%-3%或8%-15%: 3分
       <1%或>15%: 1分
       无数据: 3分

    注意: 流通市值和换手率需要从extra_data传入，
    如果没有则使用合理默认值(不惩罚)
    """
    score = 0
    details = {}

    # === A. 流通市值适中性 (最高5分) ===
    market_cap = extra_data.get('market_cap', None) if extra_data else None

    if market_cap is not None:
        cap_billion = market_cap / 1e8  # 转为亿元
        if 50 <= cap_billion <= 500:
            cap_score = 5
            details['流通市值'] = {'score': 5, 'desc': f'{cap_billion:.0f}亿(50-500亿最佳区间)'}
        elif 20 <= cap_billion < 50 or 500 < cap_billion <= 1000:
            cap_score = 3
            details['流通市值'] = {'score': 3, 'desc': f'{cap_billion:.0f}亿(次优区间)'}
        else:
            cap_score = 1
            details['流通市值'] = {'score': 1, 'desc': f'{cap_billion:.0f}亿(偏大或偏小)'}
    else:
        cap_score = 3  # 默认不惩罚
        details['流通市值'] = {'score': 3, 'desc': '未提供市值数据(默认3分)'}

    score += cap_score

    # === B. 换手率健康度 (最高5分) ===
    hsl = extra_data.get('hsl', None) if extra_data else None

    if hsl is not None:
        if 3 <= hsl <= 8:
            hsl_score = 5
            details['换手率'] = {'score': 5, 'desc': f'{hsl:.1f}%(3-8%活跃适中)'}
        elif 1 <= hsl < 3 or 8 < hsl <= 15:
            hsl_score = 3
            details['换手率'] = {'score': 3, 'desc': f'{hsl:.1f}%(偏低调或偏高)'}
        else:
            hsl_score = 1
            details['换手率'] = {'score': 1, 'desc': f'{hsl:.1f}%(过低或过高)'}
    else:
        hsl_score = 3  # 默认不惩罚
        details['换手率'] = {'score': 3, 'desc': '未提供换手率数据(默认3分)'}

    score += hsl_score

    # 钳位到10分
    score = min(score, 10)

    return score, details


# ============================================================
# 维度7: 催化与预期差 (标记，不参与自动评分)
# ============================================================

def _dim_catalyst(extra_data=None):
    """
    催化与预期差 -- 灵魂信号(未量化)

    这是八大特征中最难量化的维度，涉及:
    - 题材热点共振
    - 业绩拐点
    - 政策催化
    - 预期差(市场一致预期与实际之间的差距)

    自动评分系统中仅做标记，不参与计算。
    但对最终评级有影响: 有催化剂可降低升级门槛。

    Returns:
        dict with 'has_catalyst', 'type', 'note'
    """
    has_catalyst = extra_data.get('has_catalyst', False) if extra_data else False
    catalyst_type = extra_data.get('catalyst_type', '') if extra_data else ''
    catalyst_note = extra_data.get('catalyst_note', '') if extra_data else ''

    return {
        'has_catalyst': has_catalyst,
        'type': catalyst_type,
        'note': catalyst_note or '灵魂信号(未量化) -- 需人工判定: 题材热点/业绩拐点/政策催化',
        'score_impact': '有催化剂时评级门槛降5分',
    }


# ============================================================
# 维度8: 时间窗口 (过滤条件，通过折扣系数影响总分)
# ============================================================

def _dim_time_window(market_closes=None):
    """
    时间窗口 -- 大盘环境过滤条件

    大牛股的诞生需要合适的土壤:
    - 牛市/上升趋势: 大盘环境良好，牛股辈出
    - 震荡市: 个股独立行情可以，但概率降低
    - 熊市: 逆势牛股稀少

    返回折扣系数(0.6-1.0):
    - 上证 > MA20: 1.0(环境良好)
    - 上证 < MA20 且 > MA60: 0.8(震荡环境)
    - 上证 < MA60: 0.6(熊市环境)
    - 无大盘数据: 1.0(不惩罚)
    """
    if market_closes is None or len(market_closes) < 60:
        return 1.0, {'discount': 1.0, 'desc': '无大盘数据(不折扣)'}

    ma20 = _ma(market_closes, 20)
    ma60 = _ma(market_closes, 60)

    last_close = market_closes[-1]
    last_ma20 = _safe_get(ma20, -1)
    last_ma60 = _safe_get(ma60, -1)

    if last_ma20 is None or last_ma60 is None:
        return 1.0, {'discount': 1.0, 'desc': '大盘均线数据不足(不折扣)'}

    if last_close > last_ma20:
        discount = 1.0
        desc = f'大盘站上MA20(环境良好) 折扣={discount}'
    elif last_close > last_ma60:
        discount = 0.8
        desc = f'大盘MA20-MA60区间(震荡) 折扣={discount}'
    else:
        discount = 0.6
        desc = f'大盘跌破MA60(弱势) 折扣={discount}'

    return discount, {'discount': discount, 'desc': desc}


# ============================================================
# 阶段新鲜度 (修正系数 0.82~1.0, 2026-07-20 新增)
# 替代被移除的"技术形态"维度角色。
# 解决原N类"高分=趋势末端"偏置: 奖励刚启动/早段延伸, 惩罚过度延展的末段票。
# ============================================================

def _dim_stage_freshness(closes):
    """
    阶段新鲜度 -- 修正系数(0.82~1.0)。

    两个量化锚点:
      1) 延伸度 ext = 收盘/MA60 - 1  (价格相对中期中枢的乖离)
      2) 趋势时长 trend_age = 距 MA20 上穿 MA60(金叉)以来的交易日数

    mult 含义:
      1.0  = 阶段新鲜(早段, 刚启动/刚突破)
      0.82 = 末段延展最严重(价格远高于中枢 且/或 趋势已老)
    越接近0.82, 末段特征越强, N类总分被压得越低。
    """
    n = len(closes)
    details = {'ext_pct': None, 'trend_age': None, 'mult': 1.0, 'desc': '数据不足'}
    if n < 60:
        details['desc'] = 'K线<60日, 阶段无法判定(不修正)'
        return 1.0, details

    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    last_ma60 = _safe_get(ma60, -1)
    if last_ma60 is None or last_ma60 <= 0:
        details['desc'] = 'MA60无效(不修正)'
        return 1.0, details

    ext = closes[-1] / last_ma60 - 1.0
    details['ext_pct'] = round(ext * 100, 1)

    # 趋势时长: 最近一次 MA20 上穿 MA60 的位置
    trend_age = None
    for i in range(n - 1, 0, -1):
        a20 = ma20[i]; a60 = ma60[i]; p20 = ma20[i - 1]; p60 = ma60[i - 1]
        if a20 is None or a60 is None or p20 is None or p60 is None:
            continue
        if a20 > a60 and p20 <= p60:
            trend_age = (n - 1) - i
            break
    details['trend_age'] = trend_age

    # ---- 延伸度修正 (ext: 价格相对MA60乖离) ----
    if ext <= 0.20:
        ext_mult = 1.0
    elif ext >= 0.50:
        ext_mult = 0.82
    else:
        ext_mult = 1.0 - (ext - 0.20) / (0.50 - 0.20) * (1.0 - 0.82)

    # ---- 趋势时长修正 (仅在上升趋势内: 当前 MA20>MA60) ----
    age_mult = 1.0
    cur20 = _safe_get(ma20, -1)
    cur60 = _safe_get(ma60, -1)
    if cur20 is not None and cur60 is not None and cur20 > cur60 and trend_age is not None:
        if trend_age >= 120:
            age_mult = 0.85
        elif trend_age >= 60:
            age_mult = 0.90

    mult = min(ext_mult, age_mult)
    if mult >= 1.0:
        desc = f'阶段新鲜(延伸{ext*100:.0f}%, 趋势{trend_age}日)'
    elif mult >= 0.92:
        desc = f'阶段偏中(延伸{ext*100:.0f}%, 趋势{trend_age}日)'
    else:
        desc = f'末段延展预警(延伸{ext*100:.0f}%, 趋势{trend_age}日)'
    details['mult'] = round(mult, 3)
    details['desc'] = desc
    return mult, details


# ============================================================
# 主函数: 计算牛股判断评分
# ============================================================

def compute_niugu_score(klines,
                        all_closes_map=None,
                        market_closes=None,
                        extra_data=None):
    """
    计算N类牛股判断评分(0-100分)。

    Args:
        klines: K线数据列表, 与kline_cache格式一致
                [{date, open, close, high, low, volume, amount}, ...]
        all_closes_map: 全市场收盘价映射(用于RPS计算)
                {code: [close1, close2, ...]} 或 None
        market_closes: 大盘指数收盘价序列(用于时间窗口过滤)
                [close1, close2, ...] 或 None
        extra_data: 附加数据(筹码结构/催化标记)
                {market_cap: float, hsl: float,
                 has_catalyst: bool, catalyst_type: str, catalyst_note: str}
                或 None

    Returns:
        {
            'n_score': int,         # 最终得分(0-100，经折扣)
            'n_raw_score': int,     # 原始得分(0-100，未经折扣)
            'n_grade': str,         # 评级
            'n_grade_stars': str,   # 星级标记
            'discount': float,      # 大盘折扣系数
            'dimensions': {
                '趋势结构': {'score': int, 'max': 25, 'details': {...}},
                '量价关系': {'score': int, 'max': 20, 'details': {...}},
                '相对强弱': {'score': int, 'max': 20, 'details': {...}},
                '波动率压缩': {'score': int, 'max': 15, 'details': {...}},
                '技术形态': {'score': int, 'max': 10, 'details': {...}},
                '筹码结构': {'score': int, 'max': 10, 'details': {...}},
                '催化与预期差': {...},   # 标记
                '时间窗口': {...},       # 过滤
            },
            'flags': [str],         # 关键标记列表
            'summary': str,         # 一句话总结
        }
    """
    if not klines or len(klines) < 20:
        return {
            'n_score': 0, 'n_raw_score': 0,
            'n_grade': '不达标', 'n_grade_stars': '--',
            'discount': 1.0,
            'dimensions': {},
            'flags': ['数据不足'],
            'summary': f'K线不足({len(klines) if klines else 0}日)',
        }

    # 提取数组
    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    volumes = np.array([float(k['volume']) for k in klines])
    opens = np.array([float(k.get('open', k['close'])) for k in klines])

    n = len(closes)

    # === 逐维度计算 ===
    dim1_score, dim1_details = _dim_trend_structure(closes, highs, lows)
    dim2_score, dim2_details = _dim_volume_price(closes, volumes, opens, lows, highs)
    dim3_score, dim3_details = _dim_relative_strength(closes, all_closes_map)
    dim4_score, dim4_details = _dim_volatility_compression(closes, highs, lows)
    dim5_score, dim5_details = _dim_technical_pattern(closes, highs, lows, volumes)
    dim6_score, dim6_details = _dim_chip_structure(closes, volumes, extra_data)
    dim7_result = _dim_catalyst(extra_data)
    discount, dim8_result = _dim_time_window(market_closes)
    # 阶段新鲜度修正(替代被移除的技术形态维度角色, 2026-07-20)
    stage_mult, stage_details = _dim_stage_freshness(closes)

    # === 汇总 ===
    # 技术形态(dim5)已移除，与Kimi买点平台突破/杯柄/趋势中继重合（2026-06-30）
    # 五维原始分(上限90) -> 阶段新鲜度修正 -> 时间窗口折扣
    raw_score = dim1_score + dim2_score + dim3_score + dim4_score + dim6_score
    adjusted_raw = raw_score * stage_mult
    final_score = int(adjusted_raw * discount)

    # 催化剂降门槛: 有催化剂时评级门槛降5分
    grade_threshold_offset = -5 if dim7_result['has_catalyst'] else 0

    if final_score >= 80 + grade_threshold_offset:
        grade = '大牛股候选'
        grade_stars = '***'
    elif final_score >= 60 + grade_threshold_offset:
        grade = '强势股'
        grade_stars = '**'
    elif final_score >= 40 + grade_threshold_offset:
        grade = '潜力股'
        grade_stars = '*'
    else:
        grade = '不达标'
        grade_stars = '--'

    # 关键标记
    flags = []
    if dim1_score >= 20:
        flags.append('趋势极佳')
    if dim2_score >= 15:
        flags.append('量能配合')
    if dim3_score >= 16:
        flags.append('相对强势')
    if dim4_score >= 12:
        flags.append('波动压缩')
    if dim7_result['has_catalyst']:
        flags.append('有催化剂')
    if discount < 1.0:
        flags.append(f'大盘折扣x{discount}')
    if stage_mult < 0.9:
        flags.append('末段延展预警')
    elif stage_mult >= 1.0:
        flags.append('阶段新鲜')

    # 一句话总结
    top_dims = []
    dim_names = ['趋势结构', '量价关系', '相对强弱', '波动率压缩', '筹码结构']
    dim_scores = [dim1_score, dim2_score, dim3_score, dim4_score, dim6_score]
    dim_maxes = [25, 20, 20, 15, 10]

    for name, s, m in zip(dim_names, dim_scores, dim_maxes):
        if s >= m * 0.7:  # 得分超过满分70%的维度
            top_dims.append(name)

    if top_dims:
        summary = f'N={final_score}分 [{grade}] 强项: {",".join(top_dims)}'
    else:
        summary = f'N={final_score}分 [{grade}] 各维度均需提升'

    return {
        'n_score': final_score,
        'n_raw_score': int(adjusted_raw),
        'n_grade': grade,
        'n_grade_stars': grade_stars,
        'discount': discount,
        'dimensions': {
            '趋势结构': {'score': dim1_score, 'max': 25, 'details': dim1_details},
            '量价关系': {'score': dim2_score, 'max': 20, 'details': dim2_details},
            '相对强弱': {'score': dim3_score, 'max': 20, 'details': dim3_details},
            '波动率压缩': {'score': dim4_score, 'max': 15, 'details': dim4_details},
            '技术形态': {'score': dim5_score, 'max': 10, 'details': dim5_details},
            '筹码结构': {'score': dim6_score, 'max': 10, 'details': dim6_details},
            '阶段新鲜度': {'score': round((stage_mult - 0.82) / (1.0 - 0.82) * 15), 'max': 15,
                        'details': stage_details},
            '催化与预期差': dim7_result,
            '时间窗口': dim8_result,
        },
        'flags': flags,
        'summary': summary,
        'dimension_summary': {
            name: {'score': s, 'max': m, 'pct': round(s / m * 100) if m > 0 else 0}
            for name, s, m in zip(dim_names, dim_scores, dim_maxes)
        },
    }


# ============================================================
# 批量扫描接口(给统一扫描器用)
# ============================================================

def compute_niugu_batch(klines_map, market_closes=None, extra_data_map=None):
    """
    批量计算牛股判断评分。

    Args:
        klines_map: {code: klines_list, ...}
        market_closes: 大盘收盘价序列
        extra_data_map: {code: {market_cap, hsl, ...}, ...}

    Returns:
        {code: niugu_result, ...}
    """
    # 先收集全市场收盘价映射(用于RPS)
    all_closes_map = {}
    for code, klines in klines_map.items():
        if klines and len(klines) >= 20:
            all_closes_map[code] = np.array([float(k['close']) for k in klines])

    results = {}
    for code, klines in klines_map.items():
        extra = extra_data_map.get(code, None) if extra_data_map else None
        result = compute_niugu_score(
            klines,
            all_closes_map=all_closes_map,
            market_closes=market_closes,
            extra_data=extra,
        )
        results[code] = result

    return results


# ============================================================
# 独立运行: 对kline_cache.db中所有股票进行评分
# ============================================================

def main():
    """独立运行入口，对kline_cache.db中所有股票进行牛股判断评分"""
    import sqlite3
    import json
    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, BASE_DIR)

    db_path = os.path.join(BASE_DIR, 'kline_cache.db')
    if not os.path.exists(db_path):
        print(f'[ERROR] kline_cache.db 不存在: {db_path}')
        print('请先运行 build_cache.py 或 incremental_update.py 建立K线缓存')
        return

    from kline_cache import init_db, get_klines, get_all_codes

    conn = init_db(db_path=db_path)
    all_codes = list(get_all_codes(conn))
    print(f'K线缓存: {len(all_codes)} 只股票')

    # 加载选股池名称映射
    pool_path = os.path.join(BASE_DIR, 'stock_pool.json')
    name_map = {}
    if os.path.exists(pool_path):
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        for s in pool.get('stocks', []):
            code = str(s.get('code', ''))
            name_map[code] = s.get('name', '')

    # 加载D类实时数据(含换手率/市值)
    d_class_path = os.path.join(BASE_DIR, 'd_class_realtime.json')
    extra_data_map = {}
    if os.path.exists(d_class_path):
        with open(d_class_path, 'r', encoding='utf-8') as f:
            d_data = json.load(f)
        for code, info in d_data.get('stocks', {}).items():
            extra_data_map[code] = {
                'hsl': info.get('hsl', None),
                'market_cap': info.get('market_cap', None),
            }

    # 收集所有K线数据
    klines_map = {}
    for code in all_codes:
        klines = get_klines(conn, code, min_days=60)
        if klines and len(klines) >= 60:
            klines_map[code] = klines

    conn.close()
    print(f'有效K线(>=60日): {len(klines_map)} 只')

    # 批量计算
    print('\n正在计算牛股判断评分...')
    results = compute_niugu_batch(klines_map, extra_data_map=extra_data_map)

    # 排序输出
    ranked = sorted(results.items(), key=lambda x: x[1]['n_score'], reverse=True)

    print('\n' + '=' * 90)
    print(' 牛股判断模块 V1.0 -- 大牛股启动前八大特征量化评分')
    print('=' * 90)
    print(f"{'排名':<4} {'代码':<8} {'名称':<10} {'N分':<5} {'评级':<10} {'趋势':>4} {'量价':>4} "
          f"{'强弱':>4} {'波动':>4} {'形态':>4} {'筹码':>4} {'折扣':>4} {'标记'}")
    print('-' * 90)

    for rank, (code, r) in enumerate(ranked[:50], 1):  # 只显示Top50
        name = name_map.get(code, '---')
        d = r['dimensions']
        ds = r.get('dimension_summary', {})

        trend_s = ds.get('趋势结构', {}).get('score', 0)
        vol_s = ds.get('量价关系', {}).get('score', 0)
        rs_s = ds.get('相对强弱', {}).get('score', 0)
        vola_s = ds.get('波动率压缩', {}).get('score', 0)
        pat_s = ds.get('技术形态', {}).get('score', 0)
        chip_s = ds.get('筹码结构', {}).get('score', 0)
        discount = r.get('discount', 1.0)
        flags = '/'.join(r.get('flags', [])[:3])

        print(f"{rank:<4} {code:<8} {name:<10} {r['n_score']:<5} {r['n_grade']:<10} "
              f"{trend_s:>4} {vol_s:>4} {rs_s:>4} {vola_s:>4} {pat_s:>4} {chip_s:>4} "
              f"{'x' + str(discount) if discount < 1 else '1.0':>4} {flags}")

    # 统计
    grades = {'大牛股候选': 0, '强势股': 0, '潜力股': 0, '不达标': 0}
    for code, r in results.items():
        grades[r['n_grade']] = grades.get(r['n_grade'], 0) + 1

    print('\n' + '=' * 90)
    print(' 评级分布')
    print('=' * 90)
    for g, c in grades.items():
        bar = '#' * (c // 2)
        print(f"  {g:<10} {c:>4}只  {bar}")

    # 保存结果
    output = {
        'scan_time': __import__('datetime').datetime.now().isoformat(),
        'system': '牛股判断模块 V1.0',
        'total': len(results),
        'grades': grades,
        'results': {
            code: {
                'n_score': r['n_score'],
                'n_raw_score': r['n_raw_score'],
                'n_grade': r['n_grade'],
                'n_grade_stars': r['n_grade_stars'],
                'discount': r['discount'],
                'name': name_map.get(code, ''),
                'summary': r['summary'],
                'flags': r.get('flags', []),
                'dimension_summary': r.get('dimension_summary', {}),
            }
            for code, r in results.items()
        }
    }

    output_file = os.path.join(BASE_DIR, 'niugu_scores.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\n完整结果已保存: {output_file}')


if __name__ == '__main__':
    main()
