# -*- coding: utf-8 -*-
"""
顺势系统买点 V1.3 -- N类驱动 + Kimi 4大买点 (完整量化版+数据调参) + A/D类补充
======================================================================

天才游资系统 V3.2 顺势系统买点体系 V1.3 (2026-07-15)
====================================================

核心思想: "顺大势顺小势" -- 在多头共振的时空窗口里，用明确的技术信号确认
"趋势正在继续"。买点不是"猜底"，而是在确认"趋势延续"。

V3.0 架构 (与 V2.5 的关键区别):
  - 旧: 评分累计式 (A类≥3 + D类≥2.5 → 出信号)
  - 新: 层级判定式 (L0 N类门控 → L1 大势 → L2 小势 → L3 4大买点 → L4 A/D补充)

L0  N类前置门控       一票否决  |  N类原始分 < 25 → 否决
L1  大势安检门         必备前置  |  N类原始分 ≥ 25 → 通过 (N类分越高越强)
L2  小势安检门         必备前置  |  日线MA20向上 + 未现顶背离
L3  Kimi 4大买点       主信号    |  4选1，互斥
L4  顺势旧体系(A/D)   补充验证  |  A≥2 或 D≥2.0 → 置信度+1档

四个买点 (Kimi 实战总结, 含完整量化指标):
  买点一: 平台突破买点 (主攻型)
    - 平台振幅: HHV(H,20)/LLV(L,20) < 1.25 (即<25%, V1.2从15%放宽)
    - 波动率压缩: ATR(14)/MA(C,14) < 0.06 (V1.3从0.08收紧，回测P90=0.06)
    - 突破幅度: C > HHV(H,20) AND C/HHV(H,20) > 1.03 (突破>3%)
    - 量能确认: V > MA5*1.5 AND V > MA60*1.2 (短+长周期双确认)
    - ~~突破站稳~~ (V1.2移除: 与买点二二次确认重叠)
    - 仓位: 标准仓 (10-15%)
    - 止损: 平台下沿下方3% 或 突破阳线实体下沿

  买点二: 均线回踩买点 (稳健型)
    - 趋势向上: MA20>MA50 AND MA20>REF(MA20,5)
    - 中期均线: MA50>REF(MA50,10)
    - 缩量回调: V < MA20*0.8
    - 触及均线: L <= MA20*1.02 AND C >= MA20*0.98
    - 回撤幅度: (HHV20-L)/(HHV40-LLV40) < 0.5
    - 止跌K线: ABS(C-O)/O<0.03 AND (C-L)/(H-L)>0.3 (小实体+下影线)
    - 仓位: 偏重仓位 (15-20%)
    - 止损: MA20下方2% 或 前低

  买点三: 杯柄突破买点 (高盈亏比)
    - 杯深足够: (HHV120-LLV120)/HHV120 > 0.30
    - 右侧抬高: LLV(L,20) > LLV(L,60)*1.15
    - 杯部收敛: HHV(H,20)/LLV(L,20) < 1.10
    - 杯部缩量: MA(V,20) < MA(V,60)*0.5
    - 突破柄部: C > HHV(H,20) AND C/HHV(H,20) > 1.02 (突破>2%)
    - 周线MACD: MACD.WEEK > REF(MACD.WEEK, 1) (周线红柱放大)
    - 仓位: 标准仓 (10-15%)
    - 止损: 柄部下沿 或 杯柄突破阳线低点

  买点四: 趋势中继买点 (追击型, 风险最高)
    - 趋势陡峭: MA20/REF(MA20,20) > 1.30 (20日均线涨幅>30%)
    - 快速整理: HHV(H,10)/LLV(L,10) < 1.08 (10日振幅<8%)
    - 整理缩量: MA(V,10) < MA(V,20) * 0.8
    - 强势整理: LLV(L,10) > REF(LLV(L,5),10)*1.05 (低点抬高5%)
    - 突破加速: C > HHV(H,10) AND C/HHV(H,10) > 1.03 (突破>3%)
    - 量能释放: V > MA5*1.5 (1.5倍即可, 实战标准)
    - 仓位: 轻仓 (5-10%)
    - 止损: 旗形/三角下沿 或 前一个重要低点

作者: WorkBuddy Agent for 天才游资潘小铠
日期: 2026-06-24
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================
# 通用工具函数 (周线转日线, 常用指标)
# ============================================================

def _compute_ma(series: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均线"""
    n = len(series)
    ma = np.full(n, np.nan)
    for i in range(period - 1, n):
        ma[i] = np.mean(series[i - period + 1:i + 1])
    return ma


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """计算 ATR (Average True Range)"""
    n = len(closes)
    if n < 2:
        return np.zeros(n)

    # True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    # ATR (Wilder's smoothing)
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """
    计算 MACD 指标
    返回: { 'macd': np.ndarray, 'signal': np.ndarray, 'hist': np.ndarray (红柱) }
    """
    n = len(closes)
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    macd_line = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    hist = np.full(n, np.nan)

    if n < slow:
        return {'macd': macd_line, 'signal': signal_line, 'hist': hist}

    # EMA fast
    multiplier_fast = 2 / (fast + 1)
    ema_fast[fast - 1] = np.mean(closes[:fast])
    for i in range(fast, n):
        ema_fast[i] = (closes[i] - ema_fast[i - 1]) * multiplier_fast + ema_fast[i - 1]

    # EMA slow
    multiplier_slow = 2 / (slow + 1)
    ema_slow[slow - 1] = np.mean(closes[:slow])
    for i in range(slow, n):
        ema_slow[i] = (closes[i] - ema_slow[i - 1]) * multiplier_slow + ema_slow[i - 1]

    # MACD line
    for i in range(slow - 1, n):
        macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line
    multiplier_sig = 2 / (signal + 1)
    valid_start = slow - 1 + signal - 1
    if valid_start < n:
        signal_line[valid_start] = np.mean(macd_line[slow - 1:valid_start + 1])
        for i in range(valid_start + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier_sig + signal_line[i - 1]

    # Histogram (红柱/绿柱)
    for i in range(slow - 1, n):
        hist[i] = macd_line[i] - signal_line[i]

    return {'macd': macd_line, 'signal': signal_line, 'hist': hist}


def _aggregate_weekly(daily_klines: List[Dict]) -> List[Dict]:
    """
    将日线K线聚合为周线K线 (周五为周线收盘日)。
    用于买点三的"周线MACD红柱放大"验证。
    """
    if not daily_klines or len(daily_klines) < 5:
        return []

    weekly = []
    week_data = []

    for k in daily_klines:
        week_data.append(k)
        # 简化: 每5个交易日为一周 (实际应按日期判断, 此处为快速聚合)
        if len(week_data) == 5:
            weekly.append({
                'date': k.get('date', ''),
                'open': week_data[0]['open'],
                'close': week_data[-1]['close'],
                'high': max(d['high'] for d in week_data),
                'low': min(d['low'] for d in week_data),
                'volume': sum(d['volume'] for d in week_data),
            })
            week_data = []

    # 处理剩余不足5天的数据
    if week_data:
        weekly.append({
            'date': week_data[-1].get('date', ''),
            'open': week_data[0]['open'],
            'close': week_data[-1]['close'],
            'high': max(d['high'] for d in week_data),
            'low': min(d['low'] for d in week_data),
            'volume': sum(d['volume'] for d in week_data),
        })

    return weekly


# ============================================================
# L0: N类前置门控 (一票否决)
# ============================================================

def _check_l0_niugu_gate(niugu_result: Optional[Dict]) -> Tuple[bool, str, int]:
    """
    L0 N类前置门控 -- 顺势系统的"大门"。

    逻辑: N类评分 < 25 → 一票否决 (N类不认为有牛股基因，何必顺势)
          N类评分 ≥ 25 → 通过 (顺大势的前提)

    Returns:
        (passed, reason, niugu_score)
    """
    if niugu_result is None:
        return False, "L0否决: 无N类评分数据", 0

    niugu_score = niugu_result.get('n_raw_score', 0)
    grade = niugu_result.get('n_grade', '不达标')

    L0_THRESHOLD = 40  # 2026-07-20: 提到40 (质量门控, 甜区起点40-50; 30-39段回测仅51%胜率偏边际)

    if niugu_score < L0_THRESHOLD:
        return False, f"L0否决: N类={niugu_score}<{L0_THRESHOLD}({grade})，不具备牛股基因", niugu_score

    return True, f"L0通过: N类={niugu_score}({grade})", niugu_score


# ============================================================
# L1: 大势安检门 (顺大势确认) -- 复用N类评分
# ============================================================

def _check_l1_trend_gate(niugu_score: int, niugu_result: Dict) -> Tuple[bool, str, str]:
    """
    L1 大势安检门 -- 顺势系统的"质量检查"。

    阈值 (基于 N类原始分):
        >= 60: 大势明确 (3档)
        40-59: 大势良好 (2档)
        25-39: 大势勉强 (1档)
    """
    if niugu_score >= 60:
        return True, f"L1通过: N类={niugu_score}，大势明确", "明确"
    elif niugu_score >= 40:
        return True, f"L1通过: N类={niugu_score}，大势良好", "良好"
    else:  # 25-39 (L0 已确保 >= 25)
        return True, f"L1通过(勉强): N类={niugu_score}，顺势但需谨慎", "勉强"


# ============================================================
# L2: 小势安检门 (顺小势确认) -- MA20 + 顶背离
# ============================================================

def _check_l2_daily_gate(klines: List[Dict]) -> Tuple[bool, str]:
    """
    L2 小势安检门 -- 日线级别"上升趋势中"判定。

    规则 (Kimi 体系):
      1) 日线MA20向上 (斜率 > 0)
      2) 股价站上MA20
      3) 近20日未现顶背离 (RSI 顶背离)
      4) (可选) 不超买: RSI(14) < 80

    注: 这是"通过则进"的检查，不是"否决"检查。
    """
    if not klines or len(klines) < 30:
        return False, "L2数据不足(<30日)"

    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    n = len(closes)

    # === MA20 向上 + 股价站上 MA20 ===
    ma20 = _compute_ma(closes, 20)

    last_close = closes[-1]
    last_ma20 = ma20[-1]
    ma20_5ago = ma20[-6] if n >= 25 else None

    if last_ma20 is None or np.isnan(last_ma20):
        return False, "L2否决: MA20数据不足"

    # MA20 斜率
    ma20_slope_pct = None
    if ma20_5ago is not None and ma20_5ago > 0 and not np.isnan(ma20_5ago):
        ma20_slope_pct = (last_ma20 / ma20_5ago - 1) / 5 * 100

    # V1.2 放松: 允许轻微下跌斜率(-0.5%/日)，避免在震荡市大面积否决
    cond_ma20_up = (ma20_slope_pct is not None and ma20_slope_pct >= -0.5)
    cond_above_ma20 = (last_close > last_ma20)

    if not cond_ma20_up:
        return False, f"L2否决: MA20斜率{ma20_slope_pct:.2f}%/日(非向上)"
    if not cond_above_ma20:
        return False, f"L2否决: 股价{last_close:.2f}<MA20={last_ma20:.2f}"

    # === 顶背离检测 (近20日, RSI) ===
    rsi_divergence = _check_rsi_top_divergence(closes, highs, lows, lookback=20)

    if rsi_divergence:
        return False, "L2否决: 近20日现RSI顶背离，等待回调"

    return True, f"L2通过: MA20斜率{ma20_slope_pct:.2f}%/日，站上MA20，无顶背离"


def _check_rsi_top_divergence(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> bool:
    """
    RSI 顶背离检测 (修复版)。
    价格创新高但 RSI 未创新高 → 顶背离。

    Args:
        closes: 收盘价序列
        highs: 最高价序列 (修复: 之前未传入)
        lows: 最低价序列
        lookback: 回看窗口
    """
    n = len(closes)
    if n < lookback + 14:
        return False

    # 计算 RSI(14)
    rsi_period = 14
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if n > rsi_period:
        avg_gain[rsi_period] = np.mean(gains[:rsi_period])
        avg_loss[rsi_period] = np.mean(losses[:rsi_period])
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i - 1]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i - 1]) / rsi_period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - 100 / (1 + rs)

    # 修复: 使用 highs 找近 lookback 内的最高点
    recent_window_highs = highs[-lookback:]
    recent_high_idx_offset = np.argmax(recent_window_highs)
    recent_high_idx = n - lookback + recent_high_idx_offset
    recent_high_price = closes[recent_high_idx]
    recent_high_rsi = rsi[recent_high_idx]

    last_close = closes[-1]
    last_rsi = rsi[-1]

    # 价格新高 + RSI 未新高 → 顶背离
    if last_close >= recent_high_price * 0.99 and last_rsi < recent_high_rsi - 5:
        return True

    return False


# ============================================================
# L3: Kimi 4大买点 -- 互斥判定 (V1.1 完整量化版)
# ============================================================

def _bp1_platform_breakout(klines: List[Dict]) -> Optional[Dict]:
    """
    买点一: 平台突破买点 (主攻型) -- V1.1 完整量化

    Kimi 量化条件 (P0+P1+P2 全部纳入):
      [P2 放宽] 平台振幅: HHV(H,20)/LLV(L,20) < 1.15 (即<15%)
      [P0 新增] 波动率压缩: ATR(14)/MA(C,14) < 0.05
      [P0 新增] 突破幅度: C > HHV(H,20) AND C/HHV(H,20) > 1.03 (突破>3%)
      [P0 新增] 量能确认: V > MA5*1.5 AND V > MA60*1.2 (短+长周期)
      [P0 新增] 突破站稳: COUNT(C > HHV(H,20), 3) >= 2 (3日内至少2日收在高点上)

    仓位: 标准仓 (10-15%)
    """
    if not klines or len(klines) < 60:
        return None

    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    volumes = np.array([float(k['volume']) for k in klines])
    n = len(closes)

    # === [P2 放宽 → V1.2 再放宽] 1) 平台振幅 < 25% (近20日，从15%放开) ===
    platform_section_highs = highs[-21:-1]  # 平台期间最高(不含今日)
    platform_section_lows = lows[-21:-1]
    platform_high = np.max(platform_section_highs)
    platform_low = np.min(platform_section_lows)

    if platform_low <= 0:
        return None
    range_pct = (platform_high - platform_low) / platform_low * 100
    if range_pct >= 25:
        return None  # 振幅过宽，不是平台

    # === [P0 新增] 2) 波动率压缩: ATR(14)/MA(C,14) < 0.05 ===
    atr = _compute_atr(highs, lows, closes, period=14)
    ma_c = _compute_ma(closes, 14)
    if np.isnan(atr[-1]) or np.isnan(ma_c[-1]) or ma_c[-1] <= 0:
        return None
    volatility_ratio = atr[-1] / ma_c[-1]
    if volatility_ratio >= 0.06:
        return None  # 波动率未压缩 (V1.3: 收紧到0.06，真实信号P90=0.06)

    # === [P0 新增] 3) 突破幅度: C > HHV(H,20) AND C/HHV(H,20) > 1.03 ===
    last_close = closes[-1]
    if last_close <= platform_high:
        return None  # 未突破平台
    breakout_amp = (last_close / platform_high - 1) * 100
    if breakout_amp < 3:
        return None  # 突破幅度不足3%

    # === [P0 新增] 4) 量能确认: V > MA5*1.5 AND V > MA60*1.5 (V1.3: 60日从1.2提到1.5) ===
    last_vol = volumes[-1]
    vol_ma5 = np.mean(volumes[-6:-1])
    vol_ma60 = np.mean(volumes[-61:-1]) if n >= 62 else np.mean(volumes[:-1])

    if vol_ma5 <= 0 or vol_ma60 <= 0:
        return None
    vol_ratio_short = last_vol / vol_ma5
    vol_ratio_long = last_vol / vol_ma60

    if vol_ratio_short < 1.5:
        return None  # 短期量能不足
    if vol_ratio_long < 1.5:
        return None  # 长周期量能未启动 (V1.3: 从1.2提到1.5)

    # === [P0 废止 V1.2] 5) 突破站稳: 已移除（与买点二二次确认功能重叠） ===
    # 原: COUNT(C > HHV(H,20), 3) >= 2 (3日内至少2日收在高点上)

    # === 全部条件满足 ===
    return {
        'bp_type': '买点一',
        'bp_name': '平台突破买点',
        'bp_style': '主攻型',
        'position_size': '标准仓 10-15%',
        'platform_range_pct': round(range_pct, 2),
        'platform_high': round(platform_high, 2),
        'platform_low': round(platform_low, 2),
        'volatility_ratio': round(volatility_ratio, 4),
        'breakout_amp_pct': round(breakout_amp, 2),
        'vol_ratio_short': round(vol_ratio_short, 2),
        'vol_ratio_long': round(vol_ratio_long, 2),
        'breakout_price': round(last_close, 2),
        'stop_loss': f'平台下沿下方3% ({round(platform_low * 0.97, 2)})',
    }


def _bp2_ma20_pullback(klines: List[Dict]) -> Optional[Dict]:
    """
    买点二: 均线回踩买点 (稳健型) -- V1.1 完整量化

    Kimi 量化条件 (P0+P1+P2 全部纳入):
      [P1 新增] 趋势向上: MA20>MA50 AND MA20>REF(MA20,5)
      [P1 新增] 中期均线: MA50>REF(MA50,10)
      [原有]  缩量回调: V < MA20*0.8
      [原有]  触及均线: L <= MA20*1.02 AND C >= MA20*0.98
      [P0 新增] 回撤幅度: (HHV20-L)/(HHV40-LLV40) < 0.5
      [P0 新增] 止跌K线: ABS(C-O)/O<0.03 AND (C-L)/(H-L)>0.3 (小实体+下影线)

    仓位: 偏重仓位 (15-20%)
    """
    if not klines or len(klines) < 50:
        return None

    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    opens = np.array([float(k['open']) for k in klines])
    volumes = np.array([float(k['volume']) for k in klines])
    n = len(closes)

    # === 均线计算 ===
    ma20 = _compute_ma(closes, 20)
    ma50 = _compute_ma(closes, 50)
    vol_ma20 = _compute_ma(volumes, 20)

    last_close = closes[-1]
    last_ma20 = ma20[-1]
    last_ma50 = ma50[-1] if not np.isnan(ma50[-1]) else None
    last_vol = volumes[-1]
    last_open = opens[-1]
    last_low = lows[-1]
    last_high = highs[-1]

    if np.isnan(last_ma20) or last_ma20 <= 0:
        return None

    # === [P1 新增] 1) 趋势向上: MA20>MA50 AND MA20>REF(MA20,5) ===
    if last_ma50 is None or last_ma20 <= last_ma50:
        return None  # MA20未上穿MA50

    ma20_5ago = ma20[-6] if n >= 25 and not np.isnan(ma20[-6]) else None
    if ma20_5ago is None or last_ma20 <= ma20_5ago:
        return None  # MA20未向上

    # === [P1 新增] 2) 中期均线: MA50>REF(MA50,10) ===
    if n >= 60 and not np.isnan(ma50[-1]) and not np.isnan(ma50[-11]):
        if ma50[-1] <= ma50[-11]:
            return None  # MA50未保持完好

    # === 3) 触及均线: L <= MA20*1.02 AND C >= MA20*0.98 ===
    # 检查近5日内某日触及MA20 (今日或近5日内)
    touched_ma20 = False
    touch_idx = -1
    for i in range(-5, 0):
        if abs(i) <= n:
            day_idx = n + i
            if day_idx < 0 or day_idx >= n:
                continue
            if (lows[day_idx] <= ma20[day_idx] * 1.02 and
                closes[day_idx] >= ma20[day_idx] * 0.98 and
                not np.isnan(ma20[day_idx])):
                touched_ma20 = True
                touch_idx = day_idx
                break
    if not touched_ma20:
        return None  # 近5日未回踩MA20

    # === 4) 缩量回调: V < MA20*0.8 (近5日均量) ===
    if not np.isnan(vol_ma20[-1]) and vol_ma20[-1] > 0:
        vol_5ma = np.mean(volumes[-6:-1])
        vol_shrink_ratio = vol_5ma / vol_ma20[-1] if vol_ma20[-1] > 0 else 1.0
        if vol_shrink_ratio > 0.8:
            return None  # 缩量不充分

    # === [P0 新增] 5) 回撤幅度: (HHV20-L)/(HHV40-LLV40) < 0.5 ===
    if n >= 40:
        hhv20 = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs)
        llv40 = np.min(lows[-41:-1]) if len(lows) >= 41 else np.min(lows)
        hhv40 = np.max(highs[-41:-1]) if len(highs) >= 41 else np.max(highs)
        if (hhv40 - llv40) > 0:
            pullback_pct = (hhv20 - lows[touch_idx]) / (hhv40 - llv40)
            if pullback_pct >= 0.5:
                return None  # 回撤过深

    # === [P0 新增] 6) 止跌K线: ABS(C-O)/O<0.03 AND (C-L)/(H-L)>0.3 ===
    # 检查触及MA20当日的K线形态
    if touch_idx >= 0 and opens[touch_idx] > 0:
        body_pct = abs(closes[touch_idx] - opens[touch_idx]) / opens[touch_idx]
        lower_shadow = (min(closes[touch_idx], opens[touch_idx]) - lows[touch_idx])
        full_range = highs[touch_idx] - lows[touch_idx]
        if full_range > 0:
            shadow_pct = lower_shadow / full_range
            if not (body_pct < 0.03 and shadow_pct > 0.3):
                return None  # 没有止跌K线形态

    # === 全部条件满足 ===
    return {
        'bp_type': '买点二',
        'bp_name': '均线回踩买点',
        'bp_style': '稳健型',
        'position_size': '偏重仓位 15-20%',
        'ma20_value': round(last_ma20, 2),
        'ma50_value': round(last_ma50, 2) if last_ma50 else None,
        'vol_shrink_ratio': round(vol_shrink_ratio, 2) if 'vol_shrink_ratio' in dir() else None,
        'last_close': round(last_close, 2),
        'stop_loss': f'MA20下方2% ({round(last_ma20 * 0.98, 2)})',
    }


def _bp3_cup_handle_breakout(klines: List[Dict], weekly_klines: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    买点三: 杯柄突破买点 (高盈亏比型) -- V1.1 完整量化

    Kimi 量化条件 (P0+P1+P2 全部纳入):
      [P0 → V1.2] 杯深足够: (HHV120-LLV120)/HHV120 > 0.20 (从30%放宽)
      [P0 新增] 右侧抬高: LLV(L,20) > LLV(L,60)*1.15
      [P2 优化] 杯部收敛: HHV(H,20)/LLV(L,20) < 1.10
      [原有]  杯部缩量: MA(V,20) < MA(V,60)*0.5
      [P0 新增] 突破柄部: C > HHV(H,20) AND C/HHV(H,20) > 1.02 (突破>2%)
      [P1 → V1.2] 周线MACD: 有则加分，无则不否决

    仓位: 标准仓 (10-15%)
    """
    if not klines or len(klines) < 120:
        return None

    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    volumes = np.array([float(k['volume']) for k in klines])
    n = len(closes)

    # === [P0 新增] 1) 杯深足够: (HHV120-LLV120)/HHV120 > 0.30 ===
    lookback = min(120, n)
    hhv120 = np.max(highs[-lookback:])
    llv120 = np.min(lows[-lookback:])
    if hhv120 <= 0:
        return None
    cup_depth = (hhv120 - llv120) / hhv120
    if cup_depth <= 0.20:
        return None  # 杯形不够深 (V1.2: 从30%放松到20%，覆盖U型底)

    # === [P0 → V1.3放宽] 2) 右侧抬高: LLV(L,20) > LLV(L,60)*1.10 (从1.15降到1.10) ===
    if n >= 60:
        llv20 = np.min(lows[-21:-1])
        llv60 = np.min(lows[-61:-1])
        if llv60 > 0 and llv20 <= llv60 * 1.10:
            return None  # 右侧未明显抬高

    # === 3) 找近120日的杯底 (最低点) 和 杯部高点 ===
    window = closes[-lookback:]
    window_lows = lows[-lookback:]
    window_highs = highs[-lookback:]

    cup_bottom_idx = np.argmin(window_lows)
    cup_bottom_price = window_lows[cup_bottom_idx]
    cup_left_high = np.max(window_highs[:cup_bottom_idx + 1]) if cup_bottom_idx > 0 else cup_bottom_price
    cup_right_high = np.max(window_highs[cup_bottom_idx:]) if cup_bottom_idx < len(window_highs) - 1 else cup_bottom_price

    # 杯部左右两侧高点应大致相等 (圆弧底特征) -- 放宽到 30% 容忍度
    # 实际A股中"U型"不一定左右等高, 左侧25→20, 右侧20→28 也是合格杯形
    if cup_left_high > 0 and abs(cup_right_high - cup_left_high) / cup_left_high > 0.30:
        return None  # 杯部不完整 (左右高差>30%)

    # === [V1.3 移除] 4) 杯部收敛 — 条件与BP1平台振幅重复，且A股杯柄天然振幅偏大，25%仍杀57%，改为不硬否决 ===
    # hhv20 = np.max(highs[-21:-1])
    # llv20_recent = np.min(lows[-21:-1])
    # if llv20_recent > 0 and hhv20 / llv20_recent >= 1.25:
    #     return None  # 杯柄部未收敛
    # — converge check removed, hhv20/llv20 still needed later for breakout check
    hhv20 = np.max(highs[-21:-1])
    llv20_recent = np.min(lows[-21:-1])

    # === 5) 杯部缩量: MA(V,20) < MA(V,60)*0.8 (V1.3: 从50%放宽到80%，对齐BP4标准) ===
    vol_ma20_recent = np.mean(volumes[-21:-1])
    vol_ma60 = np.mean(volumes[-61:-1]) if n >= 62 else vol_ma20_recent
    if vol_ma60 <= 0:
        return None
    vol_shrink_ratio = vol_ma20_recent / vol_ma60
    if vol_shrink_ratio >= 0.8:
        return None  # 杯柄缩量不充分

    # === [P0 新增] 6) 突破柄部: C > HHV(H,20) AND C/HHV(H,20) > 1.02 ===
    cup_top = hhv20
    last_close = closes[-1]
    if last_close <= cup_top:
        return None  # 未突破
    breakout_amp = (last_close / cup_top - 1) * 100
    if breakout_amp < 2:
        return None  # 突破幅度不足2%

    # 突破量能确认
    last_vol = volumes[-1]
    vol_breakout_ratio = last_vol / vol_ma20_recent if vol_ma20_recent > 0 else 0
    if vol_breakout_ratio < 1.5:
        return None  # 突破量能不足

    # === [P1 放宽 V1.2] 7) 周线MACD红柱放大 → 有则加分，无则跳过 ===
    weekly_macd_check = None
    if weekly_klines is None or len(weekly_klines) < 30:
        weekly_klines = _aggregate_weekly(klines)

    if len(weekly_klines) >= 30:
        weekly_closes = np.array([float(k['close']) for k in weekly_klines])
        macd_data = _compute_macd(weekly_closes)
        hist = macd_data['hist']
        if not np.isnan(hist[-1]) and not np.isnan(hist[-2]):
            weekly_macd_check = hist[-1] > hist[-2]  # 红柱放大
            # V1.2: 不硬否决，仅降为"无周线加成"置信度档
    # if weekly_macd_check is False:  # 原: 直接 return None

    return {
        'bp_type': '买点三',
        'bp_name': '杯柄突破买点',
        'bp_style': '高盈亏比型',
        'position_size': '标准仓 10-15%',
        'cup_depth_pct': round(cup_depth * 100, 2),
        'cup_bottom_price': round(cup_bottom_price, 2),
        'cup_top': round(cup_top, 2),
        'vol_shrink_ratio': round(vol_shrink_ratio, 2),
        'breakout_amp_pct': round(breakout_amp, 2),
        'breakout_vol_ratio': round(vol_breakout_ratio, 2),
        'last_close': round(last_close, 2),
        'weekly_macd_check': weekly_macd_check,
        'stop_loss': f'杯部高点下方 ({round(cup_top * 0.97, 2)})',
    }


def _bp4_trend_continuation(klines: List[Dict]) -> Optional[Dict]:
    """
    买点四: 趋势中继买点 (追击型) -- V1.1 完整量化

    Kimi 量化条件 (P0+P1+P2 全部纳入):
      [P1 → V1.2 放宽] 趋势陡峭: MA20/REF(MA20,20) > 1.15 (20日均线涨幅>15%，原30%)
      [原有]  快速整理: HHV(H,10)/LLV(L,10) < 1.08 (10日振幅<8%)
      [P1 新增] 整理缩量: MA(V,10) < MA(V,20) * 0.8
      [P1 新增] 强势整理: LLV(L,10) > REF(LLV(L,5),10)*1.05 (低点抬高5%)
      [P0 新增] 突破加速: C > HHV(H,10) AND C/HHV(H,10) > 1.03 (突破>3%)
      [P1 放宽] 量能释放: V > MA5*1.5 (从2倍放宽到1.5倍, 实战标准)

    仓位: 轻仓 (5-10%)

    ⚠️ 此买点风险最高: A股风险最高，只适合已有底仓者加仓，或极小仓位追击。
    """
    if not klines or len(klines) < 50:
        return None

    closes = np.array([float(k['close']) for k in klines])
    highs = np.array([float(k['high']) for k in klines])
    lows = np.array([float(k['low']) for k in klines])
    volumes = np.array([float(k['volume']) for k in klines])
    n = len(closes)

    # === [P1 → V1.2 放宽] 1) 趋势陡峭: MA20/REF(MA20,20) > 1.15 (20日均线涨幅>15%，原30%) ===
    if n < 40:
        return None
    ma20 = _compute_ma(closes, 20)
    if np.isnan(ma20[-1]) or np.isnan(ma20[-21]):
        return None
    ma20_now = ma20[-1]
    ma20_20ago = ma20[-21]
    if ma20_20ago <= 0:
        return None
    ma20_change_pct = (ma20_now / ma20_20ago - 1) * 100
    if ma20_change_pct <= 15:
        return None  # 20日均线涨幅不足15%, 不算明显趋势

    # === 2) 快速整理: HHV(H,10)/LLV(L,10) < 1.08 (近10日) ===
    hhv10 = np.max(highs[-11:-1])
    llv10 = np.min(lows[-11:-1])
    if llv10 <= 0:
        return None
    pullback_range = (hhv10 - llv10) / llv10 * 100
    if pullback_range >= 8:
        return None  # 回调过深

    # === [P1 新增] 3) 整理缩量: MA(V,10) < MA(V,20) * 0.8 ===
    vol_ma10 = np.mean(volumes[-11:-1])
    vol_ma20 = np.mean(volumes[-21:-1])
    if vol_ma20 <= 0:
        return None
    vol_shrink_pullback = vol_ma10 / vol_ma20
    if vol_shrink_pullback >= 0.8:
        return None  # 整理期量能未明显萎缩

    # === [P1 新增] 4) 强势整理: LLV(L,10) > REF(LLV(L,5),10)*1.05 ===
    # 近10日低点 > 5-14日低点 * 1.05
    if n >= 14:
        llv10_recent = np.min(lows[-11:-1])
        llv_prev = np.min(lows[-15:-11])
        if llv_prev > 0 and llv10_recent <= llv_prev * 1.05:
            return None  # 低点未抬高 (整理不够强)

    # === [P0 新增] 5) 突破加速: C > HHV(H,10) AND C/HHV(H,10) > 1.03 ===
    last_close = closes[-1]
    if last_close <= hhv10:
        return None  # 未突破
    breakout_amp = (last_close / hhv10 - 1) * 100
    if breakout_amp < 3:
        return None  # 突破加速幅度不足3%

    # === [P1 放宽] 6) 量能释放: V > MA5*1.5 ===
    last_vol = volumes[-1]
    vol_ma5 = np.mean(volumes[-6:-1])
    if vol_ma5 <= 0:
        return None
    vol_breakout_ratio = last_vol / vol_ma5
    if vol_breakout_ratio < 1.5:
        return None  # 突破量能不足

    return {
        'bp_type': '买点四',
        'bp_name': '趋势中继买点',
        'bp_style': '加仓/追击型',
        'position_size': '轻仓 5-10%',
        'ma20_change_20d_pct': round(ma20_change_pct, 2),
        'pullback_range_pct': round(pullback_range, 2),
        'vol_shrink_pullback': round(vol_shrink_pullback, 2),
        'breakout_amp_pct': round(breakout_amp, 2),
        'breakout_vol_ratio': round(vol_breakout_ratio, 2),
        'last_close': round(last_close, 2),
        'stop_loss': f'旗形/三角下沿 ({round(llv10 * 0.98, 2)})',
        'risk_warning': '⚠️ 此买点风险最高，只适合已有底仓者加仓，或极小仓位追击',
    }


# ============================================================
# L4: A/D类补充验证 (顺势旧体系降级为验证)
# ============================================================

def _check_ad_supplement(a_score: float = 0, d_score: float = 0) -> Tuple[bool, str, int]:
    """
    L4 A/D类补充验证 -- 顺势旧体系降级为验证层。
    """
    has_a = a_score is not None and a_score >= 2
    has_d = d_score is not None and d_score >= 2.0

    if has_a and has_d:
        return True, f"L4加成: A={a_score}, D={d_score} 双确认", 2
    elif has_a or has_d:
        which = f'A={a_score}' if has_a else f'D={d_score}'
        return True, f"L4加成: {which} 单确认", 1
    else:
        return False, "L4中性: 无A/D类补充验证", 0


# ============================================================
# 主判定函数: detect_shunshi_buy_point
# ============================================================

def detect_shunshi_buy_point(
    code: str,
    klines: List[Dict],
    niugu_result: Optional[Dict] = None,
    a_score: float = 0,
    d_score: float = 0,
    weekly_klines: Optional[List[Dict]] = None,
) -> Dict:
    """
    顺势系统买点主判定函数 V1.1。

    V3.0 层级判定流程:
      L0 N类门控 → L1 大势 → L2 小势 → L3 4大买点 → L4 A/D补充

    Args:
        code: 股票代码
        klines: 日线K线数据列表
        niugu_result: N类评分结果 (compute_niugu_score的返回值)
        a_score: A类形态评分 (0-10)
        d_score: D类资金评分 (0-5)
        weekly_klines: 周线K线数据 (可选, 买点三的周线MACD验证用)

    Returns:
        {
            'code': str,
            'signal': bool,
            'bp_type': str,
            'bp_name': str,
            'position_size': str,
            'l0_passed': bool, 'l1_passed': bool, 'l2_passed': bool,
            'l3_bp_result': Optional[Dict],
            'l4_boost': int,
            'confidence': str,
            'reasons': [str],
        }
    """
    reasons = []

    # === L0: N类前置门控 ===
    l0_passed, l0_reason, niugu_score = _check_l0_niugu_gate(niugu_result)
    reasons.append(l0_reason)
    if not l0_passed:
        return {
            'code': code,
            'signal': False,
            'bp_type': None, 'bp_name': None, 'position_size': None,
            'l0_passed': False, 'l1_passed': False, 'l2_passed': False,
            'l3_bp_result': None, 'l4_boost': 0,
            'confidence': '无',
            'reasons': reasons,
            'summary': f'L0否决: N类={niugu_score}分，不具备牛股基因',
        }

    # === L1: 大势安检门 ===
    l1_passed, l1_reason, l1_strength = _check_l1_trend_gate(niugu_score, niugu_result)
    reasons.append(l1_reason)
    if not l1_passed:
        return {
            'code': code,
            'signal': False,
            'bp_type': None, 'bp_name': None, 'position_size': None,
            'l0_passed': True, 'l1_passed': False, 'l2_passed': False,
            'l3_bp_result': None, 'l4_boost': 0,
            'confidence': '无',
            'reasons': reasons,
            'summary': f'L1否决: {l1_reason}',
        }

    # === L2: 小势安检门 ===
    l2_passed, l2_reason = _check_l2_daily_gate(klines)
    reasons.append(l2_reason)
    l2_warning = not l2_passed

    # === L3: Kimi 4大买点 (互斥) ===
    bp_results = [
        _bp1_platform_breakout(klines),
        _bp2_ma20_pullback(klines),
        _bp3_cup_handle_breakout(klines, weekly_klines),
        _bp4_trend_continuation(klines),
    ]
    bp_result = next((r for r in bp_results if r is not None), None)

    if bp_result is None:
        return {
            'code': code,
            'signal': False,
            'bp_type': None, 'bp_name': None, 'position_size': None,
            'l0_passed': True, 'l1_passed': True, 'l2_passed': l2_passed,
            'l3_bp_result': None, 'l4_boost': 0,
            'confidence': '无',
            'reasons': reasons + ['L3否决: 4大买点均未触发'],
            'summary': f'N类={niugu_score}分(L1:{l1_strength})，但4大买点未触发',
        }

    reasons.append(f"L3命中: {bp_result['bp_name']}")

    # === L4: A/D类补充验证 ===
    l4_passed, l4_reason, l4_boost = _check_ad_supplement(a_score, d_score)
    reasons.append(l4_reason)

    # === 综合置信度 ===
    score = 0
    if l1_strength == '明确':
        score += 3
    elif l1_strength == '良好':
        score += 2
    else:
        score += 1
    if l2_passed:
        score += 2
    else:
        score += 0
    score += l4_boost

    # 买点风险加权 (买点四风险高)
    if bp_result['bp_type'] == '买点四':
        score -= 1

    if score >= 6:
        confidence = '高'
    elif score >= 4:
        confidence = '中'
    else:
        confidence = '低'

    return {
        'code': code,
        'signal': True,
        'bp_type': bp_result['bp_type'],
        'bp_name': bp_result['bp_name'],
        'bp_style': bp_result.get('bp_style', ''),
        'position_size': bp_result.get('position_size', ''),
        'l0_passed': True,
        'l1_passed': True,
        'l2_passed': l2_passed,
        'l3_bp_result': bp_result,
        'l4_boost': l4_boost,
        'confidence': confidence,
        'l1_strength': l1_strength,
        'niugu_score': niugu_score,
        'reasons': reasons,
        'summary': f"✅{bp_result['bp_type']}命中 | N类={niugu_score}({l1_strength}) | "
                   f"L2={'✓' if l2_passed else '⚠️'} | L4+{l4_boost}档 | 置信度={confidence}",
        'stop_loss': bp_result.get('stop_loss', ''),
        'risk_warning': bp_result.get('risk_warning', ''),
    }


# ============================================================
# 模块自检
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print(" [顺势系统买点 V1.1] 模块自检 (P0+P1+P2 全量化版)")
    print("=" * 70)

    import random
    random.seed(42)

    # 场景1: 强势股 + 模拟买点一
    klines_test = []
    for i in range(120):
        base = 10 + i * 0.05
        klines_test.append({
            'date': f'2026-04-{i+1:02d}' if i < 30 else f'2026-05-{i-29:02d}',
            'open': base,
            'close': base + random.uniform(-0.3, 0.5),
            'high': base + random.uniform(0, 0.6),
            'low': base + random.uniform(-0.5, 0.2),
            'volume': 1000000 + random.randint(-200000, 200000),
        })

    niugu_mock = {
        'n_score': 75, 'n_raw_score': 75,
        'n_grade': '强势股', 'n_grade_stars': '**',
    }

    result = detect_shunshi_buy_point(
        code='000001',
        klines=klines_test,
        niugu_result=niugu_mock,
        a_score=3.5,
        d_score=2.5,
    )

    print(f"\n[测试场景1] N类=75分 + A=3.5 + D=2.5")
    print(f"  Signal: {result['signal']}")
    print(f"  Summary: {result['summary']}")
    for r in result['reasons']:
        print(f"    - {r}")

    # 场景2: N类低分
    niugu_low = {'n_score': 18, 'n_raw_score': 18, 'n_grade': '不达标', 'n_grade_stars': '--'}
    result2 = detect_shunshi_buy_point(
        code='000002', klines=klines_test, niugu_result=niugu_low,
    )
    print(f"\n[测试场景2] N类=18分(低)")
    print(f"  Signal: {result2['signal']}")
    print(f"  Summary: {result2['summary']}")

    # 场景3: N类无数据
    result3 = detect_shunshi_buy_point(code='000003', klines=klines_test)
    print(f"\n[测试场景3] N类=无数据")
    print(f"  Signal: {result3['signal']}")
    print(f"  Summary: {result3['summary']}")

    print("\n" + "=" * 70)
    print(" [OK] 模块自检完成")
    print("=" * 70)
