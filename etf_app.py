#!/usr/bin/env python3
"""ETF双动量轮动 — 可视化运行界面 (Streamlit)

用法: streamlit run etf_app.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import json
import math

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the script dir is on sys.path so imports work both in dev and PyInstaller
sys.path.insert(0, str(Path(__file__).parent))

from etf_data import (DEFAULT_CONFIG, calc_indicators, load_config, load_prices,
                        load_open_prices, load_midday_prices, load_afternoon_open_prices,
                        midday_data_available)
from etf_backtrader import run_backtest_bt, position_dist_bt, STRATEGIES
from etf_grid import run_grid_backtest, GridConfig
from etf_grid_data import load_grid_data
import datetime as _dt
import json

st.set_page_config(page_title="ETF双动量轮动", layout="wide")

# 网格参数持久化（本地 JSON + URL）
GRID_CONFIG_PATH = Path(__file__).parent / "etf_grid_config.json"
MOMENTUM_CONFIG_PATH = Path(__file__).parent / "etf_momentum_config.json"

st.markdown("""
<style>
/* metric 数值不截断，缩小字号完整显示 */
[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    white-space: nowrap;
    overflow: visible !important;
    text-overflow: clip !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    white-space: nowrap;
    overflow: visible !important;
    text-overflow: clip !important;
}
/* 列之间减少间距 */
[data-testid="column"] {
    padding-left: 0.3rem !important;
    padding-right: 0.3rem !important;
}
    /* dataframe 按内容自适应宽度，超出屏幕自动滚动 */
    [data-testid="stDataFrame"] > div:first-child {
        width: fit-content !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def cached_prices(etfs: dict, group_name: str, source: str = "tencent") -> pd.DataFrame:
    return load_prices(etfs, group_name, source=source)


@st.cache_data(ttl=60)
def cached_open_prices(etfs: dict, group_name: str, source: str = "akshare") -> pd.DataFrame | None:
    return load_open_prices(etfs, group_name, source=source)




def _safe_loc(df, col, dt, fallback_prices, i):
    """Get df[col].loc[dt] safely, falling back to prev close if dt not in index."""
    if col is not None and col in df.columns and dt in df.index:
        v = df[col].loc[dt]
        if not pd.isna(v):
            return v
    # Fallback: use previous close (same as np.roll logic)
    if i > 0:
        return fallback_prices[col].iloc[i - 1]
    return np.nan


def run_backtest(prices, mode, start_date, end_date, ma_days, roc_days, min_hold=0,
                 open_prices=None, midday_prices=None, afternoon_open_prices=None,
                 delay=0, use_open_signal=False):
    """Inline backtest so the app stays self-contained.

    信号在 T 日收盘判定，T+1 执行。
    - 无特殊价格: close-to-close (信号 T日close → 执行 T日close, 同日)
    - open_prices: T+1 开盘执行 (信号 T日close → 执行 T+1日open)
    - midday + afternoon_open: 中午执行 (信号 T-1日close → 执行 T日中午)
    - delay: 信号延迟天数 (0=同日/次日, 1=额外延迟1天, 即当前旧行为)
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    etf_names = list(prices.columns)
    returns = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    COMMISSION = 0.0001
    STAMP_DUTY = 0.0005

    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    trades = 0
    trade_dates = []
    trade_details = []
    is_friday = prices.index.dayofweek == 4
    last_trade_idx = -999
    _use_open = open_prices is not None
    _use_midday = midday_prices is not None and afternoon_open_prices is not None
    _is_close = not _use_open and not _use_midday
    _first_bar_in_range = True if not use_open_signal else False

    # T日开盘：用开盘价计算 MA 和 ROC
    _ma_open = None
    _roc_open = None
    if use_open_signal and open_prices is not None:
        _ma_open, _roc_open, _ = calc_indicators(open_prices.ffill(), ma_days, roc_days)

    # signal_hist[j] = signal computed from close[j] (None before warmup)
    signal_hist: list = [None] * len(prices)
    daily_signals: list = []  # built per bar for UI signal table
    started_in_range = False

    for i in range(ma_days, len(prices)):
        dt = prices.index[i]

        # Reset position on first bar in backtest range → clean start
        if not started_in_range and start_date <= dt <= end_date:
            holding = None
            started_in_range = True
            _first_bar_in_range = not use_open_signal  # T日开盘首根有效

        # ── Step 1: compute signal ──
        should_check = True if mode == "daily" else is_friday[i]
        if should_check:
            above = {}
            for name in etf_names:
                if use_open_signal and open_prices is not None and name in open_prices.columns:
                    # T日开盘：用当日开盘价。若 open 没这一天数据，回退到 close
                    px = _safe_loc(open_prices, name, dt, prices, i)
                    # MA/ROC 用 .loc[dt]，避免 index 不对齐
                    if dt in _ma_open.index and dt in _roc_open.index:
                        ma = _ma_open[name].loc[dt] if not pd.isna(_ma_open[name].loc[dt]) else np.nan
                        roc = _roc_open[name].loc[dt] if not pd.isna(_roc_open[name].loc[dt]) else np.nan
                    else:
                        # 当日没有 open 指标，回退到 close
                        px, ma, roc = np.nan, np.nan, np.nan
                else:
                    px = prices[name].iloc[i]
                    ma = ma60[name].iloc[i]
                    roc = roc20[name].iloc[i]
                if not pd.isna(ma) and not pd.isna(px) and px > ma and not pd.isna(roc):
                    above[name] = roc
            signal_hist[i] = max(above, key=above.get) if above else None
        else:
            signal_hist[i] = signal_hist[i - 1] if i > ma_days else None

        # ── Step 2: determine effective signal ──
        if use_open_signal:
            src = i  # 当日开盘信号，当日执行
        elif _is_close:
            src = i - delay
        else:
            src = i - 1 - delay
        effective_signal = signal_hist[src] if src >= ma_days else None

        # ── Step 3: execute ──
        skip_first = _first_bar_in_range and not _is_close  # MOO/Midday: stale signal on first bar
        _first_bar_in_range = False
        if not skip_first and effective_signal != holding and i - last_trade_idx >= min_hold:
            if _use_midday and i > 0:
                # Midday execution: morning(old) + afternoon(new)
                mid_dt = midday_prices.index[midday_prices.index <= dt]
                aft_dt = afternoon_open_prices.index[afternoon_open_prices.index <= dt]
                if len(mid_dt) > 0 and len(aft_dt) > 0:
                    mk = mid_dt[-1]; ak = aft_dt[-1]
                    mid_ok = holding is None or (holding in midday_prices.columns and mk in midday_prices.index)
                    aft_ok = effective_signal is None or (effective_signal in afternoon_open_prices.columns and ak in afternoon_open_prices.index)
                    if (holding is None or mid_ok) and (effective_signal is None or aft_ok):
                        if holding is not None:
                            prev_c = prices[holding].iloc[i - 1]
                            mid_px = midday_prices[holding].loc[mk]
                            if not pd.isna(prev_c) and not pd.isna(mid_px) and prev_c > 0:
                                strat_ret.iloc[i] = mid_px / prev_c - 1
                            strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                        if effective_signal is not None:
                            strat_ret.iloc[i] -= COMMISSION
                        new_h = effective_signal
                        if new_h is not None:
                            aft_o = afternoon_open_prices[new_h].loc[ak]
                            day_c = prices[new_h].iloc[i]
                            if not pd.isna(aft_o) and not pd.isna(day_c) and aft_o > 0:
                                afternoon_ret = day_c / aft_o - 1
                                strat_ret.iloc[i] = (1 + strat_ret.iloc[i]) * (1 + afternoon_ret) - 1
                    else:
                        if holding is not None:
                            r = returns[holding].iloc[i]
                            strat_ret.iloc[i] = (r if not pd.isna(r) else 0.0) - COMMISSION - STAMP_DUTY
                        if effective_signal is not None:
                            strat_ret.iloc[i] -= COMMISSION
                else:
                    if holding is not None:
                        r = returns[holding].iloc[i]
                        strat_ret.iloc[i] = (r if not pd.isna(r) else 0.0) - COMMISSION - STAMP_DUTY
                    if effective_signal is not None:
                        strat_ret.iloc[i] -= COMMISSION
            elif _use_open and i > 0:
                # T+1 open execution
                # Use .loc[dt] (not .iloc[i]) because open_prices may have different row count
                if holding is not None:
                    prev_c = prices[holding].iloc[i - 1]
                    today_open_old = _safe_loc(open_prices, holding, dt, prices, i)
                    if not pd.isna(prev_c) and not pd.isna(today_open_old) and prev_c > 0:
                        strat_ret.iloc[i] = today_open_old / prev_c - 1
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if effective_signal is not None:
                    strat_ret.iloc[i] -= COMMISSION
                new_h = effective_signal
                if new_h is not None:
                    o = _safe_loc(open_prices, new_h, dt, prices, i)
                    c = prices[new_h].iloc[i]
                    if not pd.isna(o) and not pd.isna(c) and o > 0:
                        strat_ret.iloc[i] = (1 + strat_ret.iloc[i]) * (1 + c / o - 1) - 1
            else:
                # Close-to-close (signal T日close → execute T日close, same day)
                if holding is not None:
                    r = returns[holding].iloc[i]
                    strat_ret.iloc[i] = (r if not pd.isna(r) else 0.0) - COMMISSION - STAMP_DUTY
                if effective_signal is not None:
                    strat_ret.iloc[i] -= COMMISSION

            trades += 1
            trade_dates.append(dt)
            trade_details.append((dt, holding, effective_signal))
            last_trade_idx = i
            holding = effective_signal

        elif holding is not None:
            r = returns[holding].iloc[i]
            strat_ret.iloc[i] = r if not pd.isna(r) else 0.0

        # Build daily signal record (after execution, holding reflects current position)
        sig_record = {'_dt': dt}
        for name in etf_names:
            if use_open_signal and _roc_open is not None and name in _roc_open.columns and dt in _roc_open.index:
                roc_v = _roc_open[name].loc[dt]
            else:
                roc_v = roc20[name].iloc[i]
            sig_record[name] = float(roc_v) if not pd.isna(roc_v) else None
        sig_record['holding'] = holding  # post-execution holding
        daily_signals.append(sig_record)

    trim = (prices.index >= start_date) & (prices.index <= end_date)
    ret = strat_ret[trim]
    bench_ret = returns[trim].mean(axis=1)
    nav = (1 + ret).cumprod()
    bench_nav = (1 + bench_ret).cumprod()
    filtered_details = [(dt, h, nh) for dt, h, nh in trade_details
                        if start_date <= dt <= end_date]
    filtered_dates = [t[0] for t in filtered_details]
    filtered_signals = [s for s in daily_signals
                        if start_date <= s['_dt'] <= end_date]
    return nav, bench_nav, ret, bench_ret, len(filtered_details), filtered_dates, filtered_details, filtered_signals


def calc_metrics(nav, ret):
    r = ret.dropna()
    if len(r) < 1:
        return {}
    total = nav.iloc[-1] - 1
    ann = (1 + total) ** (252 / max(len(r), 1)) - 1 if total > -1 else total
    vol = r.std() * (252 ** 0.5) if len(r) >= 2 else 0.0
    sharpe = (ann - 0.03) / vol if vol > 0 else (0.0 if ann <= 0.03 else float('inf'))
    dd_series = nav / nav.cummax() - 1
    dd = dd_series.min()
    calmar = ann / abs(dd) if dd != 0 and ann > 0 else 0
    max_loss = (nav - 1).min()
    max_loss_dt = nav.idxmin()
    underwater_days = int((nav < 1).sum())
    holding_days = len(r)
    # longest consecutive period below initial capital (NAV < 1.0)
    longest_loss_days = 0
    longest_loss_start = None
    longest_loss_end = None
    cur_start = None
    cur_len = 0
    for dt, val in nav.items():
        if val < 1.0:
            if cur_start is None:
                cur_start = dt
            cur_len += 1
        else:
            if cur_len > longest_loss_days:
                longest_loss_days = cur_len
                longest_loss_start = cur_start
                longest_loss_end = nav.index[nav.index.get_loc(dt) - 1]
            cur_start = None
            cur_len = 0
    if cur_len > longest_loss_days:
        longest_loss_days = cur_len
        longest_loss_start = cur_start
        longest_loss_end = nav.index[-1]
    loss_range = f"{longest_loss_start.strftime('%Y-%m-%d')} ~ {longest_loss_end.strftime('%Y-%m-%d')}" if longest_loss_start else "N/A"
    return {"累计收益": total, "年化收益": ann, "年化波动": vol, "夏普比率": sharpe,
            "最大回撤": dd, "卡尔玛比率": calmar, "最大亏损": max_loss,
            "最大亏损日期": max_loss_dt, "水下天数": underwater_days, "持有天数": holding_days,
            "最长亏损持续": longest_loss_days, "最长亏损区间": loss_range}


def _nav_one_backtest(prices, daily_ret, ma60, roc20, etf_names, start_date, end_date, mode, min_hold, ma_days):
    """Run one backtest pass, return final NAV. etf_names = active ETF pool.
    Uses additive commission (matching run_backtest) for exact comparability."""
    is_friday = prices.index.dayofweek == 4
    COMMISSION = 0.0001
    STAMP_DUTY = 0.0005
    strat_ret = pd.Series(0.0, index=prices.index)
    holding = None
    last_trade_idx = -999
    for i in range(ma_days, len(prices)):
        dt = prices.index[i]
        if holding is not None and dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date):
            r = daily_ret[holding].iloc[i]
            if not pd.isna(r):
                strat_ret.iloc[i] = r
        should_check = True if mode == "daily" else is_friday[i]
        if should_check and (min_hold <= 0 or i - last_trade_idx >= min_hold):
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not pd.isna(ma) and px > ma and not pd.isna(roc):
                    above[name] = roc
            new_holding = max(above, key=above.get) if above else None
            if new_holding is not None and new_holding != holding:
                last_trade_idx = i
            if new_holding != holding:
                if holding is not None:
                    strat_ret.iloc[i] -= COMMISSION + STAMP_DUTY
                if new_holding is not None:
                    strat_ret.iloc[i] -= COMMISSION
            holding = new_holding
    trim = (prices.index >= pd.Timestamp(start_date)) & (prices.index <= pd.Timestamp(end_date))
    r = strat_ret[trim].dropna()
    return (1 + r).prod() if len(r) > 0 else 1.0


def position_dist(prices, start_date, end_date, mode, ma_days, roc_days, min_hold=0):
    """返回 (持有天数dict, 买入次数dict, 收益占比dict, 持有期累计收益dict, 上涨天数占比dict)
    收益占比 = 各ETF持有期间的对数收益 / 总对数收益，加总=100%，正=赚钱负=亏钱
    持有期累计收益 = 持有该ETF期间的累计收益率
    [v2: buys only counted in_range]"""
    etf_names = list(prices.columns)
    daily_ret = prices.pct_change(fill_method=None)
    ma60, roc20, _ = calc_indicators(prices, ma_days, roc_days)
    is_friday = prices.index.dayofweek == 4
    COMMISSION_RATE = 0.0001 + 0.0005  # 万1佣金 + 万5印花税（卖出），简化为双向万3
    days = {n: 0 for n in etf_names}
    days["CASH"] = 0
    buys = {n: 0 for n in etf_names}
    nav = {n: 1.0 for n in etf_names}
    pos_days = {n: 0 for n in etf_names}
    hold_days_for_wr = {n: 0 for n in etf_names}
    log_ret = {n: 0.0 for n in etf_names}
    log_ret["CASH"] = 0.0
    holding = None
    last_trade_idx = -999
    first_in_range = True
    for i in range(ma_days, len(prices)):
        dt = prices.index[i]
        in_range = dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date)
        if in_range:
            if first_in_range and holding is not None:
                buys[holding] += 1  # initial position counts as a buy
            first_in_range = False
            h = holding or "CASH"
            days[h] += 1
            if h != "CASH":
                r = daily_ret[h].iloc[i]
                if not pd.isna(r):
                    nav[h] *= (1 + r)
                    hold_days_for_wr[h] += 1
                    if r > 0:
                        pos_days[h] += 1
                    log_ret[h] += math.log(1 + r)
        should_check = True if mode == "daily" else is_friday[i]
        if should_check and (min_hold <= 0 or i - last_trade_idx >= min_hold):
            above = {}
            for name in etf_names:
                px = prices[name].iloc[i]
                ma = ma60[name].iloc[i]
                roc = roc20[name].iloc[i]
                if not pd.isna(ma) and px > ma and not pd.isna(roc):
                    above[name] = roc
            new_holding = max(above, key=above.get) if above else None
            if new_holding is not None and new_holding != holding:
                if in_range:
                    buys[new_holding] += 1
                last_trade_idx = i
                # 佣金从当天持仓的 log return 扣除
                if in_range and holding is not None:
                    log_ret[holding] += math.log(1 - COMMISSION_RATE)
            holding = new_holding

    total_log = sum(log_ret.values())
    contrib = {}
    for name in etf_names:
        contrib[name] = log_ret[name] / total_log if abs(total_log) > 1e-12 else 0.0
    contrib["CASH"] = 0.0

    cum_ret = {}
    for k in etf_names:
        cum_ret[k] = nav[k] - 1.0 if days[k] > 0 else 0.0
    cum_ret["CASH"] = 0.0

    win_rate = {}
    for k in etf_names:
        win_rate[k] = pos_days[k] / hold_days_for_wr[k] if hold_days_for_wr[k] > 0 else 0.0
    win_rate["CASH"] = 0.0

    return days, buys, contrib, cum_ret, win_rate


def yearly_returns(ret):
    result = {}
    for yr in sorted(set(ret.index.year)):
        mask = ret.index.year == yr
        if mask.sum() > 10:
            result[yr] = (1 + ret[mask]).prod() - 1
    return result


def trade_win_rate(ret, trade_details, prices):
    """Per-trade win rate: each closed position = one observation.
    Win if exit price > entry price. Standard industry metric."""
    if not trade_details:
        return 0.0
    entry_prices = {}
    wins = 0
    total = 0
    for dt, sold, bought in trade_details:
        if sold is not None and sold in entry_prices:
            exit_px = prices[sold].get(dt, np.nan) if dt in prices.index else np.nan
            if not np.isnan(exit_px):
                total += 1
                if exit_px > entry_prices[sold]:
                    wins += 1
            del entry_prices[sold]
        if bought is not None:
            entry_px = prices[bought].get(dt, np.nan) if dt in prices.index else np.nan
            if not np.isnan(entry_px):
                entry_prices[bought] = entry_px
    last_dt = ret.index[-1]
    for etf, entry_px in entry_prices.items():
        exit_px = prices[etf].get(last_dt, np.nan) if last_dt in prices.index else np.nan
        if not np.isnan(exit_px):
            total += 1
            if exit_px > entry_px:
                wins += 1
    return wins / total if total > 0 else 0.0


def grid_search(prices, modes, start, end, ma_values, roc_values, progress_bar,
                open_prices=None, midday_prices=None, afternoon_open_prices=None,
                delay=0, use_open_signal=False):
    """网格搜索最优MA/ROC，返回所有结果DataFrame"""
    import itertools

    rows = []
    total = len(ma_values) * len(roc_values) * len(modes)
    done = 0
    for ma, roc in itertools.product(ma_values, roc_values):
        for mode in modes:
            nav, bnav, ret, bret, trades, trade_dates, trade_details, _ = run_backtest(
                prices, mode, start, end, ma, roc,
                open_prices=open_prices,
                midday_prices=midday_prices,
                afternoon_open_prices=afternoon_open_prices,
                delay=delay, use_open_signal=use_open_signal)
            m = calc_metrics(nav, ret)
            wr = trade_win_rate(ret, trade_details, prices)
            rows.append({
                "MA": ma, "ROC": roc, "模式": mode,
                "累计收益": m.get("累计收益", 0),
                "年化收益": m.get("年化收益", 0),
                "夏普比率": m.get("夏普比率", 0),
                "最大回撤": m.get("最大回撤", 0),
                "最大亏损": m.get("最大亏损", 0),
                "水下天数": m.get("水下天数", 0),
                "持有天数": m.get("持有天数", 0),
                "卡尔玛比率": m.get("卡尔玛比率", 0),
                "胜率": wr,
                "交易次数": trades,
            })
            done += 1
            progress_bar.progress(done / total,
                                  text=f"MA={ma} ROC={roc} {mode}  ({done}/{total})")
    return pd.DataFrame(rows)


def build_optimizer_heatmap(df, metric):
    """Plotly热力图"""
    import plotly.graph_objects as go

    modes = df["模式"].unique()
    figs = {}
    for mode in modes:
        sub = df[df["模式"] == mode].pivot_table(index="MA", columns="ROC", values=metric)
        if sub.empty:
            continue
        fig = go.Figure(data=go.Heatmap(
            z=sub.values,
            x=list(sub.columns),
            y=list(sub.index),
            colorscale="RdYlGn",
            text=np.vectorize(lambda v: f"{v:.2f}" if abs(v) < 10 else f"{v:.1%}")(sub.values),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="MA=%{y}<br>ROC=%{x}<br>%{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{METRIC_LABELS.get(metric, metric)} ({mode})",
            xaxis_title="ROC 天数", yaxis_title="MA 天数",
            height=500,
        )
        figs[mode] = fig
    return figs


METRIC_LABELS = {
    "累计收益": "累计收益", "年化收益": "年化收益", "夏普比率": "夏普比率",
    "最大回撤": "最大回撤", "最大亏损": "最大亏损", "水下天数": "水下天数",
    "卡尔玛比率": "卡尔玛比率", "胜率": "胜率",
}

# Percentage columns in optimizer results
_OPT_PCT_COLS = {"累计收益", "年化收益", "最大回撤", "最大亏损", "胜率"}
_OPT_FLOAT_COLS = {"夏普比率", "卡尔玛比率"}
_OPT_INT_COLS = {"水下天数", "持有天数", "交易次数"}


def _fmt_optimize_table(df: pd.DataFrame) -> None:
    """In-place format optimizer result columns for display."""
    for c in _OPT_PCT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.1%}")
    for c in _OPT_FLOAT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.2f}")
    for c in _OPT_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)


def build_plotly_fig(prices, etf_codes, modes_data, start, end):
    """Reuse the same Plotly logic as etf_backtest.plot_interactive but embedded."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    strategy_colors = {"daily": "#2196F3", "friday": "#FF9800"}
    cmp_colors = {"daily": "#64B5F6", "friday": "#FFB74D"}
    etf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.05,
        subplot_titles=("净值曲线", "回撤"),
    )

    nav0 = prices[prices.index >= start].copy()
    etf_color_map = {}
    for i, name in enumerate(prices.columns):
        etf_nav = (1 + nav0[name].pct_change(fill_method=None).fillna(0)).cumprod()
        code = etf_codes.get(name, "")
        label = f"持有{name} ({code})" if code else f"持有{name}"
        color = etf_colors[i % len(etf_colors)]
        etf_color_map[name] = color
        fig.add_trace(go.Scatter(
            x=etf_nav.index, y=etf_nav.values, mode="lines",
            name=label, line=dict(color=color, width=1.5), opacity=0.7,
        ), row=1, col=1)

    bench_added = False
    for mode, (nav, bnav, trade_dates, trade_details) in modes_data.items():
        color = strategy_colors.get(mode, "black")

        cum_ret = nav - 1
        ret_customdata = [(d.strftime("%Y-%m-%d"), r) for d, r in zip(nav.index, cum_ret)]
        hovertemplate = f"策略({mode}) " + "%{customdata[0]} 净值%{y:.3f} 累计%{customdata[1]:+.2%}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, mode="lines",
            name=f"策略({mode})",
            line=dict(color=color, width=2.5),
            customdata=ret_customdata,
            hovertemplate=hovertemplate,
        ), row=1, col=1)

        # Underwater
        uw = nav.copy()
        uw[uw >= 1] = pd.NA
        if uw.notna().any():
            fig.add_trace(go.Scatter(
                x=nav.index, y=uw.values, mode="lines",
                name=f"水下({mode})",
                line=dict(color="#DC2626", width=3),
                hovertemplate=f"水下({mode}) " + "%{customdata[0]} 净值%{y:.3f}<extra></extra>",
                customdata=ret_customdata,
            ), row=1, col=1)

        # Trade markers
        if trade_details:
            valid = [(dt, frm, to) for dt, frm, to in trade_details if dt in nav.index]
            if valid:
                dates = [dt for dt, _, _ in valid]
                navs_v = [nav.loc[dt] for dt in dates]
                marker_colors = [etf_color_map.get(to, "#999") for _, _, to in valid]
                hover_texts = []
                for dt, frm, to in valid:
                    dt_str = dt.strftime("%Y-%m-%d")
                    if frm is None:
                        hover_texts.append(f"{dt_str}<br>买入 <b>{to}</b>")
                    elif to is None:
                        hover_texts.append(f"{dt_str}<br>卖出 <b>{frm}</b> → <b>空仓</b>")
                    else:
                        hover_texts.append(f"{dt_str}<br>卖出 <b>{frm}</b> → 买入 <b>{to}</b>")
                fig.add_trace(go.Scatter(
                    x=dates, y=navs_v, mode="markers",
                    name=f"调仓({mode})",
                    marker=dict(color=marker_colors, size=10, symbol="diamond",
                               line=dict(color="white", width=1)),
                    hovertemplate="%{customdata} 净值%{y:.3f}<extra></extra>",
                    customdata=hover_texts,
                ), row=1, col=1)

        # Drawdown
        dd = nav / nav.cummax() - 1
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines",
            name=f"回撤({mode})",
            line=dict(color=color, width=1),
            fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
            showlegend=False,
        ), row=2, col=1)

        if not bench_added:
            fig.add_trace(go.Scatter(
                x=bnav.index, y=bnav.values, mode="lines",
                name="等权基准", line=dict(color="#666", width=1.2, dash="dot"),
            ), row=1, col=1)
            bench_added = True

    fig.update_xaxes(range=[start, end], row=1, col=1)
    fig.update_xaxes(range=[start, end], row=2, col=1)
    fig.update_yaxes(title_text="累计净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        height=650,
    )
    return fig


def signal_for_date(prices, target_date, ma_days, roc_days):
    ma60, roc20, day_chg = calc_indicators(prices, ma_days, roc_days)
    dt = pd.Timestamp(target_date)
    if dt not in prices.index:
        available = prices.index[prices.index <= dt]
        if len(available) == 0:
            return None, None, None
        dt = available[-1]

    rows = []
    qualified = {}
    for name in prices.columns:
        px = float(prices[name].loc[dt])
        ma = float(ma60[name].loc[dt]) if not pd.isna(ma60[name].loc[dt]) else float("nan")
        roc = float(roc20[name].loc[dt]) if not pd.isna(roc20[name].loc[dt]) else float("nan")
        chg = float(day_chg[name].loc[dt]) if not pd.isna(day_chg[name].loc[dt]) else float("nan")
        rows.append({"ETF": name, "收盘价": px, "当日涨幅": chg, f"{ma_days}日均线": ma, f"{roc_days}日涨幅": roc})
        if not pd.isna(ma) and px > ma and not pd.isna(roc):
            qualified[name] = roc

    df = pd.DataFrame(rows)
    best = max(qualified, key=qualified.get) if qualified else None
    return best, df, dt


# ── Session state ──────────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()
if st.session_state.pop("_rerun_sidebar", False):
    st.rerun()

cfg = st.session_state.cfg

# ── 页面模式切换 ─────────────────────────────────────────
_mode = st.sidebar.radio("模式", ["双动量轮动", "网格交易"], horizontal=True, key="app_mode")

# ── Sidebar ──────────────────────────────────────────────

if _mode == "网格交易":
    # ═══════════════════════════════════════════════════════
    # 网格交易参数（支持 URL query params 持久化）
    # ═══════════════════════════════════════════════════════
    gq = st.query_params
    # 从 JSON 配置文件加载默认值（URL params 优先级更高）
    _grid_file_cfg = {}
    if GRID_CONFIG_PATH.exists():
        try:
            _grid_file_cfg = json.loads(GRID_CONFIG_PATH.read_text())
        except Exception:
            _grid_file_cfg = {}
    def _grid_def(k: str, default):
        """先从 URL 读，再从 JSON 读，最后用默认值"""
        if k in gq and gq[k] not in ("NaT", ""):
            return gq[k]
        if k in _grid_file_cfg:
            return _grid_file_cfg[k]
        return default

    st.sidebar.header("📊 网格参数")

    # 标的列表（可扩展）
    if "grid_symbols" not in st.session_state:
        st.session_state.grid_symbols = [
            ("510050", "上证50"), ("510300", "沪深300"), ("510500", "中证500"),
            ("588000", "科创50"), ("159915", "创业板"), ("159949", "创业板50"),
            ("159740", "纳指ETF"), ("513100", "纳指"), ("159941", "纳指"),
            ("513520", "日经ETF"), ("513500", "标普500"),
            ("512890", "红利低波"), ("515080", "红利低波"),
            ("518880", "黄金ETF"), ("159985", "豆粕ETF"),
            ("511380", "可转债"), ("511260", "十年国债"),
        ]
    # 从 session_state 恢复自定义标的
    grid_sym_list = [f"{sym} ({name})" for sym, name in st.session_state.grid_symbols]
    default_sym_idx = 0
    saved_sym = _grid_def("g_sym", "510050 (上证50)")
    if saved_sym in grid_sym_list:
        default_sym_idx = grid_sym_list.index(saved_sym)

    grid_sym_sel = st.sidebar.selectbox("标的", grid_sym_list, index=default_sym_idx,
                                        key="g_sym_sel",
                                        help="输入文字搜索，支持键盘上下选择")
    grid_symbol = grid_sym_sel.split("(")[0].strip()

    with st.sidebar.expander("📝 管理标的", expanded=False):
        # 添加
        st.markdown("**添加**")
        ac1, ac2 = st.columns(2)
        new_code = ac1.text_input("代码", "", key="g_new_code", max_chars=10, label_visibility="collapsed")
        new_name = ac2.text_input("名称", "", key="g_new_name", max_chars=20, label_visibility="collapsed")
        if st.button("➕ 添加", key="g_add_btn", use_container_width=True):
            cs, ns = new_code.strip(), new_name.strip()
            if cs and ns:
                if any(s == cs for s, _ in st.session_state.grid_symbols):
                    st.warning("该代码已存在")
                else:
                    st.session_state.grid_symbols.append((cs, ns))
                    st.rerun()
        # 编辑/删除：下拉选择要操作的标的
        st.markdown("**编辑/删除**")
        edit_sel = st.selectbox("选择标的", grid_sym_list, key="g_edit_sel",
                                label_visibility="collapsed")
        ec1, ec2, ec3 = st.columns([2, 1, 1])
        edit_new_name = ec1.text_input("新名称", value=edit_sel.split("(")[1].rstrip(")") if "(" in edit_sel else "",
                                       key="g_edit_name", label_visibility="collapsed")
        if ec2.button("✏️ 改名", key="g_rename_btn", use_container_width=True):
            ec = edit_sel.split("(")[0].strip()
            for i, (code, _) in enumerate(st.session_state.grid_symbols):
                if code == ec:
                    st.session_state.grid_symbols[i] = (code, edit_new_name.strip())
                    st.rerun()
        if ec3.button("🗑️ 删除", key="g_del_btn", use_container_width=True):
            ec = edit_sel.split("(")[0].strip()
            for i, (code, _) in enumerate(st.session_state.grid_symbols):
                if code == ec:
                    st.session_state.grid_symbols.pop(i)
                    st.rerun()

    grid_period = st.sidebar.selectbox("K线粒度", ["daily", "1", "5", "15", "30", "60"],
                                       index=["daily","1","5","15","30","60"].index(_grid_def("g_period", "daily")),
                                       format_func=lambda x: f"{x} 分钟" if x != "daily" else "日线",
                                       key="g_period_sel")
    # 数据源：分钟线只有 EM，日线有 akshare(Sina) 和 EM
    if grid_period == "daily":
        grid_source = st.sidebar.selectbox("数据源", ["akshare", "em"],
                                           index=["akshare","em"].index(_grid_def("g_src", "akshare")),
                                           format_func=lambda x: {"akshare": "AKShare (Sina)",
                                                                  "em": "东方财富 (EM)"}[x],
                                           key="g_src_sel")
    else:
        grid_source = "em"
        st.sidebar.caption("分钟线仅支持东方财富 (EM) 数据源")
    grid_type = st.sidebar.selectbox("网格类型", ["arithmetic", "geometric", "volatility"],
                                     index={"arithmetic": 0, "geometric": 1, "volatility": 2}.get(
                                         _grid_def("g_type", "arithmetic"), 0),
                                     format_func=lambda x: {"arithmetic": "等差网格",
                                                            "geometric": "等比网格",
                                                            "volatility": "ATR 动态网格"}[x],
                                     key="g_type_sel")
    grid_use_step = st.sidebar.checkbox("按步长设置", value=_grid_def("g_step", "0") != "0",
                                         help="勾选后按价差/百分比自动计算网格线数，更灵活")
    if grid_use_step:
        step_label = "每格价差" if grid_type == "arithmetic" else "每格百分比(%)"
        step_fmt = "%.4f" if grid_type == "arithmetic" else "%.2f"
        step_default = 0.05 if grid_type == "arithmetic" else 1.0
        step_saved = float(_grid_def("g_step", str(step_default)))
        if step_saved < 0.001:
            step_saved = step_default
        grid_step = st.sidebar.number_input(step_label, 0.001, 100.0,
                                             step_saved,
                                             step=0.01, format=step_fmt, key="g_step_inp")
        grid_n = 0
    else:
        grid_step = 0.0
        grid_n = st.sidebar.slider("网格线数", 4, 50, int(_grid_def("g_n", "10")),
                                   help="价格区间内划分网格数量", key="g_n_slider")
    grid_amount = st.sidebar.number_input("每格金额", 1000, 100000,
                                          int(_grid_def("g_amt", "10000")), step=1000,
                                          help="每条网格线触发时买入/卖出的金额",
                                          key="g_amt_inp")
    grid_max_pos = st.sidebar.slider("最大持仓格数", 1, 30,
                                     int(_grid_def("g_maxp", "10")),
                                     help="最多同时持有几个网格的仓位",
                                     key="g_maxp_slider")
    grid_init_shares = st.sidebar.number_input("初始底仓(股)", 0, 1000000,
                                                int(_grid_def("g_init", "0")), step=1000,
                                                help="回测开始时已持有的持仓数量",
                                                key="g_init_inp")
    grid_capital = st.sidebar.number_input("总本金", 0, 10000000,
                                            int(_grid_def("g_cap", "0")), step=10000,
                                            help="总资金。0=自动按每格金额×最大持仓格数计算",
                                            key="g_cap_inp")
    grid_base_price = st.sidebar.number_input("基准价", 0.0, 10000.0,
                                               float(_grid_def("g_bp", "0")), step=0.01,
                                               format="%.2f",
                                               help="网格中心价格。设为0则用回测首日收盘价，也可手动指定",
                                               key="g_bp_inp")

    sb_date_col1_g, sb_date_col2_g = st.sidebar.columns(2)
    with sb_date_col1_g:
        grid_start = st.date_input("开始", value=pd.Timestamp(_grid_def("g_sd", "2026-01-01")),
                                    key="gs_start", format="YYYY-MM-DD",
                                    max_value=pd.Timestamp.today())
    with sb_date_col2_g:
        grid_end = st.date_input("结束", value=pd.Timestamp(_grid_def("g_ed", pd.Timestamp.today().strftime("%Y-%m-%d"))),
                                  key="gs_end", format="YYYY-MM-DD",
                                  max_value=pd.Timestamp.today())

    comm = st.sidebar.number_input("佣金率", 0.0, 0.01, float(_grid_def("g_comm", "0.0003")),
                                   step=0.0001, format="%.4f",
                                   help="ETF 万1~万3（0.0001~0.0003），A股印花税万5只收卖出",
                                   key="g_comm_inp")
    slip = st.sidebar.number_input("滑点", 0.0, 0.01, float(_grid_def("g_slip", "0.001")),
                                   step=0.0005, format="%.4f",
                                   help="成交价与网格线之间的偏差。流动性好的ETF设0.001（0.1%）",
                                   key="g_slip_inp")

    # 持久化到 URL query params
    st.query_params.update({
        "g_sym": grid_sym_sel, "g_period": grid_period, "g_src": grid_source,
        "g_type": grid_type, "g_step": str(grid_step), "g_n": str(grid_n),
        "g_amt": str(grid_amount), "g_maxp": str(grid_max_pos),
        "g_init": str(grid_init_shares), "g_cap": str(grid_capital),
        "g_bp": str(grid_base_price),
        "g_sd": str(grid_start), "g_ed": str(grid_end),
        "g_comm": str(comm), "g_slip": str(slip),
    })

    run_grid_btn = st.sidebar.button("🚀 运行网格回测", type="primary", width='stretch')
    if st.sidebar.button("💾 保存配置", width='stretch',
                         help="保存当前参数到 etf_grid_config.json，远程部署时也适用"):
        config_data = {
            "g_sym": grid_sym_sel, "g_period": grid_period, "g_src": grid_source,
            "g_type": grid_type, "g_step": str(grid_step), "g_n": str(grid_n),
            "g_amt": str(grid_amount), "g_maxp": str(grid_max_pos),
            "g_init": str(grid_init_shares), "g_cap": str(grid_capital),
            "g_pauto": "1" if grid_price_auto else "0",
            "g_pl": str(grid_pl), "g_ph": str(grid_ph),
            "g_sd": str(grid_start), "g_ed": str(grid_end),
            "g_comm": str(comm), "g_slip": str(slip),
        }
        GRID_CONFIG_PATH.write_text(json.dumps(config_data, ensure_ascii=False, indent=2))
        st.sidebar.success("✅ 配置已保存")

    # ═══════════════════════════════════════════════════════
    # 网格交易主界面
    # ═══════════════════════════════════════════════════════
    if run_grid_btn:
        import plotly.graph_objects as go
        with st.spinner(f"加载 {grid_symbol} 分钟数据..."):
            df = load_grid_data(grid_symbol, period=grid_period,
                                start_date=str(grid_start), end_date=str(grid_end),
                                source=grid_source)

        if len(df) == 0:
            st.error("❌ 未获取到数据，请检查标的代码或网络")
        else:
            with st.spinner("运行网格回测..."):
                trades, metrics, engine = run_grid_backtest(
                    grid_symbol, df, grid_type=grid_type,
                    step_value=grid_step, amount_per_grid=grid_amount,
                    max_positions=grid_max_pos,
                    initial_capital=grid_capital,
                    initial_shares=grid_init_shares,
                    base_price=grid_base_price,
                    commission=comm, slippage=slip,
                )

            from etf_grid import _is_t0
            is_t0 = _is_t0(grid_symbol)
            st.subheader(f"网格回测: {grid_symbol}  |  {grid_start} ~ {grid_end}  |  {'T+0' if is_t0 else 'T+1'}")

            # 指标卡片
            total_cap = grid_capital if grid_capital > 0 else grid_n * grid_amount if grid_n > 0 else grid_amount * grid_max_pos
            last_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0
            pos_value = metrics['持仓份额'] * last_close
            total_asset = metrics['剩余现金'] + pos_value
            mcols = st.columns(8)
            mcols[0].metric("总收益", f"{metrics['总收益']:.3%}")
            mcols[1].metric("初始现金", f"{total_cap:,.0f}")
            mcols[2].metric("总资产", f"{total_asset:,.0f}")
            mcols[3].metric("持仓金额", f"{pos_value:,.0f}")
            mcols[4].metric("买入次数", metrics.get("买入次数", 0))
            mcols[5].metric("卖出次数", metrics.get("卖出次数", 0))
            mcols[6].metric("胜率", f"{metrics['胜率']:.1%}")
            mcols[7].metric("最大回撤", f"{metrics['最大回撤']:.3%}")

            # 交易明细
            st.divider()
            st.markdown("### 📋 交易明细")
            if trades:
                trade_rows = []
                for t in trades:
                    trade_rows.append({
                        "时间": t.datetime.strftime("%m-%d %H:%M"),
                        "方向": "🟢 买入" if t.side == "buy" else "🔴 卖出",
                        "价格": f"{t.price:.4f}",
                        "金额": f"{t.amount:.0f}",
                        "份额": t.quantity,
                    })
                df_trades = pd.DataFrame(trade_rows)
                st.dataframe(df_trades, hide_index=True, width='stretch', height=300)


            # 净值图
            st.divider()
            st.markdown("### 📈 净值曲线")
            st.caption("策略总资产÷初始本金。起点=1.0(100%)，>1=盈利，<1=亏损。每笔下买单消耗现金，每笔卖单回收现金+利润，曲线反映账户整体价值变化")
            nav_series = engine.get_nav_series(df)
            if len(nav_series) > 1:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=nav_series.index, y=nav_series.values,
                    mode='lines', name='网格策略',
                    line=dict(color='#2196F3', width=2)
                ))
                final = nav_series.iloc[-1]
                fig.add_hline(y=1.0, line_color='gray', line_dash='dot', opacity=0.5)
                fig.add_hline(y=final, line_color='#2196F3', line_dash='dash', opacity=0.3)
                fig.update_layout(height=400, template='plotly_white',
                                  yaxis_tickformat='.2%',
                                  title=f'{grid_symbol} 网格策略净值 (最终: {final:.2%})')
                st.plotly_chart(fig, width='stretch')

            # 价格+K线图
            st.divider()
            st.markdown("### 📊 价格走势 + 网格线")
            if len(df) > 0:
                daily = df.resample('D').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last'
                }).dropna()
                fig2 = go.Figure()
                fig2.add_trace(go.Candlestick(
                    x=daily.index,
                    open=daily['open'], high=daily['high'],
                    low=daily['low'], close=daily['close'],
                    name=grid_symbol,
                    increasing_line_color='#E53935', decreasing_line_color='#43A047',
                ))
                # 网格线（灰度 >0=显示，0=隐藏，方便排查其他横线干扰）
                # 网格线已移除，保留 K 线 + B/S 标记
                # 买卖标记：圆点在实际成交价（对齐网格线），B/S 在当天极值外侧
                if trades:
                    buy_dates, buy_ps, sell_dates, sell_ps = [], [], [], []
                    b_text_y, s_text_y = [], []
                    daily_high = daily['high'].to_dict()
                    daily_low = daily['low'].to_dict()
                    for t in trades:
                        d = t.datetime.strftime("%Y-%m-%d")
                        ts = pd.Timestamp(d)
                        if t.side == "buy":
                            buy_dates.append(d); buy_ps.append(t.price)
                            low_px = daily_low.get(ts, t.price)
                            b_text_y.append(low_px * 0.995)
                        else:
                            sell_dates.append(d); sell_ps.append(t.price)
                            high_px = daily_high.get(ts, t.price)
                            s_text_y.append(high_px * 1.005)
                    # 实际成交价小圆点（B=红，S=蓝）
                    fig2.add_trace(go.Scatter(
                        x=buy_dates, y=buy_ps, mode='markers',
                        marker=dict(symbol='circle', size=8, color='#D32F2F', line=dict(color='white', width=1)),
                        name='买入', hovertemplate='买入 %{y:.3f}<extra></extra>', showlegend=False
                    ))
                    fig2.add_trace(go.Scatter(
                        x=sell_dates, y=sell_ps, mode='markers',
                        marker=dict(symbol='circle', size=8, color='#1976D2', line=dict(color='white', width=1)),
                        name='卖出', hovertemplate='卖出 %{y:.3f}<extra></extra>', showlegend=False
                    ))
                    # B/S 文字（B=红在K线下方，S=蓝在K线上方）
                    fig2.add_trace(go.Scatter(
                        x=buy_dates, y=b_text_y, mode='text',
                        text=['B']*len(buy_dates),
                        textfont=dict(color='#D32F2F', size=12, family='Arial Black'),
                        name='买入', hovertemplate='B %{y:.3f}<extra></extra>'
                    ))
                    fig2.add_trace(go.Scatter(
                        x=sell_dates, y=s_text_y, mode='text',
                        text=['S']*len(sell_dates),
                        textfont=dict(color='#1976D2', size=12, family='Arial Black'),
                        name='卖出', hovertemplate='S %{y:.3f}<extra></extra>'
                    ))
                fig2.update_layout(height=600, template='plotly_white',
                                   title=f'{grid_symbol} K 线 + 网格线')
                fig2.update_yaxes(showgrid=False)
                st.plotly_chart(fig2, width='stretch')

    st.stop()  # 网格模式下不执行动量逻辑


# ── 双动量轮动 ──────────────────────────────────────────
st.sidebar.header("📊 回测参数")

# ── Restore from URL query params (survives browser refresh) ──
qp = st.query_params
# 从 JSON 配置文件加载默认值（URL params 优先级更高）
_momentum_file_cfg = {}
if MOMENTUM_CONFIG_PATH.exists():
    try: _momentum_file_cfg = json.loads(MOMENTUM_CONFIG_PATH.read_text())
    except Exception: pass
_qp = lambda k, d: qp[k] if k in qp and qp[k] not in ("NaT", "") else (_momentum_file_cfg.get(k, d) if k in _momentum_file_cfg else d)


# Group selector + config button
col1, col2 = st.sidebar.columns([3, 1])
group_names = list(cfg["groups"].keys())
_default_group = "红纳创黄C" if "红纳创黄C" in group_names else group_names[0]
sel_group = col1.selectbox("组合", group_names,
                         index=group_names.index(_qp("g", _default_group)) if _qp("g", _default_group) in group_names else 0,
                         key="group_sel_v4")
with col2:
    st.write(" ")
    if st.button("⚙️", help="管理组合", width='stretch'):
        st.session_state.show_config = not st.session_state.get("show_config", False)

# Group config expander
if st.session_state.get("show_config", False):
    with st.sidebar.expander("组合管理", expanded=True):
        import json
        raw = json.dumps(cfg["groups"], ensure_ascii=False, indent=2)
        edited = st.text_area(
            "直接编辑JSON", raw, height=300, key="cfg_json",
            help="格式: {\"组合名\": {\"ETF名\": \"代码\", ...}}",
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 保存", type="primary", width='stretch'):
            try:
                parsed = json.loads(edited)
                cfg["groups"] = parsed
                with open(DEFAULT_CONFIG, "w") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                st.success("已保存")
            except json.JSONDecodeError as e:
                st.error(f"JSON格式错误: {e}")
        if c2.button("↩ 撤销", width='stretch'):
            st.session_state.pop("cfg_json", None)
            st.rerun()

# Re-sync group list after edits
group_names = list(cfg["groups"].keys())
if sel_group not in group_names:
    sel_group = "红纳创黄C" if "红纳创黄C" in group_names else group_names[0]

_start_val = pd.Timestamp(_qp("start", "2025-04-30"))
_end_val = pd.Timestamp(_qp("end", datetime.today().strftime("%Y-%m-%d")))

st.sidebar.markdown("**回测日期**")
sb_date_col1, sb_date_col2 = st.sidebar.columns(2)
with sb_date_col1:
    start_date = st.date_input("开始", value=_start_val, key="sd_start", format="YYYY-MM-DD",
                               max_value=pd.Timestamp.today())
    start_date = pd.Timestamp(start_date)
with sb_date_col2:
    end_date = st.date_input("结束", value=_end_val, key="sd_end", format="YYYY-MM-DD",
                             max_value=pd.Timestamp.today())
    end_date = pd.Timestamp(end_date)
mode = st.sidebar.radio("调仓模式", ["daily", "friday", "both"], horizontal=True,
                        index=["daily","friday","both"].index(_qp("mode", "daily")),
                        format_func=lambda x: {"daily": "每日", "friday": "周五", "both": "两者"}[x], key="sb_mode")
source = st.sidebar.selectbox("数据源", ["tencent", "akshare", "sina", "em"],
                              index=["tencent","akshare","sina","em"].index(_qp("src", "akshare")),
                              format_func=lambda x: {"tencent": "腾讯财经", "akshare": "AKShare(Sina+Tencent)", "sina": "AKShare(Sina纯源)", "em": "东方财富(EM)"}[x], key="sb_source")
source_hint = {"tencent": "⚠️ 仅约800交易日（~3年）", "akshare": "✅ 全历史(Sina) + 近3年(Tencent) 拼接", "sina": "✅ 纯Sina全历史(2011起)，最稳定，不含前复权", "em": "✅ 前复权+开盘价（东方财富，不稳定）"}
st.sidebar.caption(source_hint[source])
ma_days = st.sidebar.slider("MA 均线天数", 10, 200, int(_qp("ma", "60")), step=5, key="sb_ma")
roc_days = st.sidebar.slider("ROC 动量天数", 5, 120, int(_qp("roc", "20")), step=5, key="sb_roc")
delay = st.sidebar.slider("信号延迟 (天)", 0, 5, int(_qp("delay", "0")), step=1, key="sb_delay",
    help="0=当日收盘出信号即执行(收盘)或T+1开盘执行(开盘)。1=额外延迟1天(旧行为)")
compare_all = st.sidebar.checkbox("对比所有组合", value=False,
    help="同时回测所有已配置组合，并排对比关键指标")
run_btn = st.sidebar.button("🚀 开始回测", type="primary", width='stretch')
exec_timing = st.sidebar.selectbox("执行时机",
    ["T+1收盘", "T+1开盘", "T日开盘", "中午→下午"],
    index=0,
    format_func=lambda x: {
        "T+1开盘": "T+1 开盘执行（T日信号+T+1日开盘买卖）",
        "T+1收盘": "当日收盘执行（T日信号+T日收盘买卖）",
        "T日开盘": "T日开盘执行（T日开盘信号+T日开盘买卖）",
        "中午→下午": "中午信号→下午调仓（需60分钟K线）",
    }[x],
    key="sb_exec",
    help="T+1收盘=T日15:00收盘出信号+收盘价换仓 | T+1开盘=T日收盘出信号+T+1日开盘价换仓 | T日开盘=T日开盘价出信号+开盘价换仓")
_bt_disabled = (exec_timing in ("中午→下午", "T日开盘"))
if _bt_disabled:
    st.sidebar.info("中午/T日开盘执行模式下 Backtrader 不可用，使用手写引擎。")
use_backtrader = st.sidebar.checkbox("使用 Backtrader 引擎",
    value=False if _bt_disabled else True,
    disabled=_bt_disabled,
    help="勾选使用 backtrader 专业回测引擎，取消使用手写回测")
strategy = st.sidebar.selectbox("策略", list(STRATEGIES.keys()),
    format_func=lambda x: {"momentum": "动量轮动", "rsi": "RSI均值回归", "bb": "布林带均值回归", "macd": "MACD趋势跟随", "mom_rsi": "动量+RSI过滤", "mom_bb": "动量+布林带过滤", "vol_weighted": "波动率加权", "dual_lookback": "双周期动量", "trend_strength": "趋势确认动量", "stop_loss": "动量+移动止损", "moc": "MOC(收盘执行)", "moo": "MOO(开盘执行)"}[x],
    key="sb_strategy",
    disabled=not use_backtrader,
    help="仅 Backtrader 引擎支持多策略")

# ── Persist to URL query params (survives F5 refresh) ──
st.query_params.update({
    "g": sel_group, "start": str(start_date), "end": str(end_date),
    "mode": mode, "src": source, "ma": str(ma_days), "roc": str(roc_days),
    "stg": strategy,
    "delay": str(delay),
})

if st.sidebar.button("💾 保存动量配置", width='stretch',
                     help="保存当前参数到 etf_momentum_config.json，远程部署适用"):
    config_data = {
        "g": sel_group, "start": str(start_date), "end": str(end_date),
        "mode": mode, "src": source, "ma": str(ma_days), "roc": str(roc_days),
        "stg": strategy, "delay": str(delay),
    }
    MOMENTUM_CONFIG_PATH.write_text(json.dumps(config_data, ensure_ascii=False, indent=2))
    st.sidebar.success("✅ 动量配置已保存")

st.sidebar.divider()
st.sidebar.header("🔍 参数遍历")
optimize = st.sidebar.checkbox("开启 (搜索最优MA×ROC配置)", value=False)
if optimize:
    opt_ma_step = st.sidebar.slider("MA 步长", 5, 40, 10, help="MA 遍历步长（范围 10~200）")
    opt_roc_step = st.sidebar.slider("ROC 步长", 5, 40, 10, help="ROC 遍历步长（范围 5~120）")
    opt_metric = st.sidebar.selectbox("排序指标", ["夏普比率", "卡尔玛比率", "年化收益", "胜率", "最大亏损", "水下天数"])
    opt_top = st.sidebar.number_input("显示前N组", 5, 100, 20, step=5)

st.sidebar.divider()
st.sidebar.header("🔄 数据管理")

# 显示当前数据源缓存时间（st_mtime 是 UTC 时间戳，转为本地时间）
_cache_path = Path(__file__).parent / f"etf_prices_{source}.csv"
_cache_open_path = Path(__file__).parent / f"etf_prices_{source}_open.csv"
if _cache_path.exists():
    from datetime import datetime as _dt_lib
    _mtime = pd.Timestamp(_dt_lib.fromtimestamp(_cache_path.stat().st_mtime))
    _now = pd.Timestamp.now()
    _ago = int((_now - _mtime).total_seconds() / 60)
    if _ago < 60:
        _time_str = f"{_ago} 分钟前"
    elif _ago < 1440:
        _time_str = f"{_ago // 60} 小时前"
    else:
        _time_str = f"{_ago // 1440} 天前"
    _has_open = _cache_open_path.exists()
    st.sidebar.caption(f"最近拉取: {_mtime.strftime('%m-%d %H:%M')} ({_time_str})" + (" 📊含开盘" if _has_open else ""))
else:
    st.sidebar.caption("📦 暂无缓存，运行回测后自动拉取")

col_ref1, col_ref2 = st.sidebar.columns(2)
with col_ref1:
    if st.sidebar.button("🧹 清除缓存", width='stretch', help="删除本地CSV缓存，下次回测重新拉取"):
        import glob
        cache_dir = Path(__file__).parent
        patterns = ["etf_prices_*.csv", "etf_prices_*.csv.bak"]
        deleted = []
        for pat in patterns:
            for f in cache_dir.glob(pat):
                f.unlink()
                deleted.append(f.name)
        st.cache_data.clear()
        if deleted:
            st.sidebar.success(f"已清除 {len(deleted)} 个缓存文件")
        else:
            st.sidebar.info("无缓存文件需要清除")
        st.rerun()
with col_ref2:
    if st.sidebar.button("🔄 实时刷新", type="primary", width='stretch', help="强制实时拉取最新数据（失败自动回退到旧缓存）"):
        import shutil
        etfs = cfg["groups"][sel_group]
        cache_dir = Path(__file__).parent
        cache_file = cache_dir / f"etf_prices_{source}.csv"
        cache_open = cache_dir / f"etf_prices_{source}_open.csv"
        # 备份旧文件（用于拉取失败时回退）
        bak_file = cache_dir / f"etf_prices_{source}.csv.bak"
        bak_open = cache_dir / f"etf_prices_{source}_open.csv.bak"
        has_bak = False
        if cache_file.exists():
            shutil.copy2(cache_file, bak_file)
            has_bak = True
        if cache_open.exists():
            shutil.copy2(cache_open, bak_open)
        # 删除缓存 → 强制 load_prices 重新拉取
        cache_file.unlink(missing_ok=True)
        cache_open.unlink(missing_ok=True)
        st.cache_data.clear()
        with st.spinner(f"正在实时拉取 {source} 数据..."):
            ok = False
            try:
                _ = cached_prices(etfs, sel_group, source=source)
                _ = cached_open_prices(etfs, sel_group, source=source)
                ok = True
                st.sidebar.success("✅ 数据已实时拉取完成")
            except Exception as e:
                st.sidebar.error(f"❌ 拉取失败: {e}")
            finally:
                if not ok and has_bak:
                    # 拉取失败 → 恢复备份
                    shutil.copy2(bak_file, cache_file)
                bak_file.unlink(missing_ok=True)
                bak_open.unlink(missing_ok=True)
        st.rerun()

st.sidebar.divider()
st.sidebar.header("📡 每日信号")
sig_date = st.sidebar.date_input("查询日期", pd.Timestamp.today(), key="sig_date")
sig_btn = st.sidebar.button("🔍 查询信号", width='stretch')

with st.sidebar.expander("⏰ 实操指南"):
    st.caption(
        "**信号计算**：每个交易日下午15:00收盘后运行信号查询，使用当日收盘价计算指标。\n\n"
        "**交易执行**：下一交易日开盘（9:30）按信号调仓。"
        "回测建模 T+1 执行（信号日收盘→次交易日开盘调仓），使用真实开盘价计算滑点。\n\n"
        "**数据检查**：信号查询会自动检测数据新鲜度，"
        "若最新数据日期≠今天，说明数据源尚未更新，请等待30-60分钟后重试。"
    )

# ── Main area ────────────────────────────────────────────
if run_btn:
    etfs = cfg["groups"][sel_group]
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    lookback = (pd.Timestamp(start_str) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")

    with st.spinner("加载数据 & 运行回测..."):
        prices_full = cached_prices(etfs, sel_group, source=source)
        prices_full = prices_full[prices_full.index >= lookback]
        open_full = cached_open_prices(etfs, sel_group, source=source)
        if open_full is not None:
            open_full = open_full[open_full.index >= lookback]
        # Execution timing configuration
        _use_open_signal = (exec_timing == "T日开盘")  # T日开盘信号
        _exec_open = open_full if exec_timing in ("T+1开盘", "T日开盘") else None
        _midday_prices = None
        _afternoon_open_prices = None
        _use_midday = (exec_timing == "中午→下午")
        if _use_midday:
            _midday_prices = load_midday_prices(etfs)
            _afternoon_open_prices = load_afternoon_open_prices(etfs)
            if _midday_prices is not None and _afternoon_open_prices is not None \
                    and len(_midday_prices.columns) >= len(etfs):
                _midday_prices = _midday_prices[_midday_prices.index >= lookback]
                _afternoon_open_prices = _afternoon_open_prices[_afternoon_open_prices.index >= lookback]
                st.sidebar.success(
                    f"中午数据: {_midday_prices.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{_midday_prices.index[-1].strftime('%Y-%m-%d')}"
                )
            else:
                st.sidebar.warning("中午数据不完整，回退到 T+1 开盘执行")
                _use_midday = False
                _exec_open = open_full
        etf_codes = {name: code for name, code in etfs.items()}

        # Compute actual backtest start: when all ETFs have data + MA warmup
        first_valid_by_etf = {name: prices_full[name].first_valid_index() for name in etfs}
        latest_start = max(d for d in first_valid_by_etf.values() if d is not None)
        warmup_start = prices_full.index[min(ma_days, len(prices_full) - 1)]
        actual_start = max(latest_start, warmup_start, pd.Timestamp(start_str))
        actual_start_str = actual_start.strftime("%Y-%m-%d")

        if actual_start_str != start_str:
            st.info(f"⏱ 回测起始日已从 {start_str} 调整为 {actual_start_str}（等待所有ETF均有数据）")

        modes_to_run = ["daily", "friday"] if mode == "both" else [mode]
        modes_data = {}
        all_metrics = {}
        daily_signals_by_mode = {}

        for m in modes_to_run:
            if use_backtrader:
                bt_mode = 'moc' if _exec_open is None else 'moo'
                nav, bnav, ret, bret, trades, trade_dates, trade_details, daily_signals = \
                    run_backtest_bt(prices_full, m, actual_start_str, end_str, ma_days, roc_days,
                                    strategy=bt_mode, open_prices=_exec_open,
                                    exec_mode=bt_mode, delay=delay)
            else:
                nav, bnav, ret, bret, trades, trade_dates, trade_details, daily_signals = \
                    run_backtest(prices_full, m, actual_start_str, end_str, ma_days, roc_days,
                                 open_prices=_exec_open,
                                 midday_prices=_midday_prices,
                                 afternoon_open_prices=_afternoon_open_prices,
                                 delay=delay, use_open_signal=_use_open_signal)
            metrics_dict = calc_metrics(nav, ret)
            bench_metrics = calc_metrics(bnav, bret)
            all_metrics[m] = (metrics_dict, bench_metrics, trades, ret, nav, bnav)
            modes_data[m] = (nav, bnav, trade_dates, trade_details)
            daily_signals_by_mode[m] = daily_signals

        # 触发侧边栏时间戳刷新（数据可能被重新拉取）
        st.session_state["_rerun_sidebar"] = True

        st.subheader(f"回测结果: {sel_group}  |  {actual_start_str} ~ {end_str}")

    # ── 1. 回测指标 ──
    for m in modes_to_run:
        mm, bm, trades, ret, nav, bnav = all_metrics[m]
        trade_dates = modes_data[m][2]
        trade_details = modes_data[m][3]
        buys = sum(1 for t in trade_details if t[2] is not None)
        sells = sum(1 for t in trade_details if t[1] is not None)
        wr = trade_win_rate(ret, trade_details, prices_full)
        st.markdown(f"### 📊 {m.upper()} 策略指标")

        def render_metrics(mm, trades, wr, buys, sells, metric_keys, cols_per_row=6):
            """Render metrics in rows, cols_per_row per row. No hard column count."""
            for row_start in range(0, len(metric_keys), cols_per_row):
                row_keys = metric_keys[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for ci, key in enumerate(row_keys):
                    if key == "买入次数":
                        cols[ci].metric(key, f"{buys}", help=metric_help.get(key))
                    elif key == "卖出次数":
                        cols[ci].metric(key, f"{sells}", help=metric_help.get(key))
                    elif key in ("持有天数", "水下天数", "最长亏损持续"):
                        cols[ci].metric(key, f"{int(mm.get(key, 0))}", help=metric_help.get(key))
                    elif key == "最长亏损区间":
                        cols[ci].metric(key, mm.get(key, "N/A"), help=metric_help.get(key))
                    elif key == "最大亏损日期":
                        dt_val = mm.get("最大亏损日期")
                        dt_valid = dt_val if dt_val is not None and not pd.isna(dt_val) else None
                        cols[ci].metric(key, dt_valid.strftime("%Y-%m-%d") if dt_valid else "N/A", help=metric_help.get(key))
                    elif key == "胜率":
                        cols[ci].metric(key, f"{wr:.0%}", help=metric_help.get(key))
                    elif key in ("夏普比率", "卡尔玛比率"):
                        cols[ci].metric(key, f"{mm.get(key, 0):.2f}", help=metric_help.get(key))
                    else:
                        cols[ci].metric(key, f"{mm.get(key, 0):.3%}", help=metric_help.get(key))
            # pad empty columns so layout doesn't shift
            for ci in range(len(row_keys), cols_per_row):
                cols[ci].markdown("")

        metric_help = {
            "累计收益": "策略总收益率 = (最终净值 - 1) × 100%",
            "年化收益": "年化复合收益率，按 252 个交易日折算",
            "夏普比率": "(年化收益 - 无风险利率 3%) / 年化波动率，衡量风险调整后收益",
            "最大回撤": "策略净值从峰值到谷底的最大跌幅（峰值不一定是1.0）",
            "最大亏损": "策略净值相对本金(1.0)的最大亏损，衡量实际亏本金额度。下方显示最大亏损发生日期",
            "最大亏损日期": "最大亏损发生的具体交易日",
            "持有天数": "回测区间内的有效交易日总数",
            "水下天数": "净值低于1.0（处于亏损状态）的交易日数",
            "最长亏损持续": "净值连续低于本金(1.0)的最大交易日数（最长亏损持续期）",
            "最长亏损区间": "最长连续亏损的起止日期",
            "卡尔玛比率": "年化收益 / |最大回撤|，衡量回撤调整后收益",
            "买入次数": "策略发出的买入信号次数（买入某ETF）",
            "卖出次数": "策略发出的卖出信号次数（卖出某ETF）",
            "胜率": "获胜交易数÷总交易数，每笔买入→卖出记一次",
        }
        metric_keys = ["累计收益", "年化收益", "夏普比率", "最大回撤", "最大亏损",
                       "最大亏损日期", "水下天数", "最长亏损持续", "最长亏损区间",
                       "持有天数", "卡尔玛比率", "买入次数", "卖出次数", "胜率"]
        render_metrics(mm, trades, wr, buys, sells, metric_keys)
        pos_fn = position_dist_bt if use_backtrader else position_dist
        pos_args = (prices_full, actual_start_str, end_str, m, ma_days, roc_days)
        if use_backtrader:
            pos_kwargs = dict(strategy=bt_mode)
            pos_kwargs['open_prices'] = _exec_open
            pos_kwargs['min_hold'] = 0
        else:
            pos_kwargs = {}
        pos_days, pos_buys, pos_contrib, pos_cum, pos_wr = pos_fn(*pos_args, **pos_kwargs)
        total = sum(pos_days.values())
        if total == 0:
            st.warning("无持仓数据，跳过ETF分布展示")
        st.markdown(f"### 🗂️ {m.upper()} 持仓分布")
        pos_rows = []
        for k in sorted(pos_days.keys(), key=lambda x: -pos_days[x]):
            d = pos_days[k]
            b = pos_buys.get(k, 0)
            ct = pos_contrib.get(k, 0)
            cr = pos_cum.get(k, 0)
            wr = pos_wr.get(k, 0)
            pos_rows.append({"ETF": k, "持有天数": d,
                             "占比": f"{d/total:.0%}" if total > 0 else "N/A",
                             "买入次数": b,
                             "收益占比": f"{ct:+.3%}", "持有期累计收益": f"{cr:+.3%}", "上涨天数占比": f"{wr:.0%}"})
        st.dataframe(pd.DataFrame(pos_rows), hide_index=True, width='content')
        st.caption("收益占比=各ETF对数收益÷总对数收益(加总=100%) | 持有期累计收益=∏(1+r)-1 | 上涨天数占比=上涨天数÷持有天数")

        # Yearly returns
        yr = yearly_returns(ret)
        if len(yr) > 1:
            yr_badges = [f"` {y}: {r:+.1%} `" for y, r in yr.items()]
            st.markdown(f"### 📅 {m.upper()} 逐年收益  " + "  ".join(yr_badges))

    # ── 2. 净值曲线 ──
    st.divider()
    st.markdown("## 📈 净值曲线")
    st.plotly_chart(
        build_plotly_fig(prices_full, etf_codes, modes_data, actual_start_str, end_str),
        width='stretch',
    )

    # ── 3. ETF 价格数据 ──
    st.divider()
    st.markdown("## 💹 ETF 价格数据")
    st.markdown("**📊 ETF 真实价格**")

    # Data freshness warning
    last_data_date = prices_full.index[-1]
    today = pd.Timestamp.now().normalize()
    if last_data_date < today - pd.Timedelta(days=1):
        st.warning(f"⚠️ 数据最新日期为 {last_data_date.strftime('%Y-%m-%d')}，可能不是最新。点左侧「刷新数据缓存」获取最新数据。")

    # Check for recent data gaps per ETF (e.g. EastMoney partial failure)
    recent = prices_full.iloc[-10:]
    gap_etfs = []
    for name in prices_full.columns:
        if recent[name].isna().all():
            last_valid = prices_full[name].dropna().index[-1]
            gap_etfs.append((name, last_valid))
    if gap_etfs:
        gap_list = "、".join(f"{n}(最近: {d.strftime('%m-%d')})" for n, d in gap_etfs)
        st.warning(f"⚠️ 以下ETF近期无数据，已用前值填充(ffill)：{gap_list}。建议切换数据源或刷新缓存。")

    # Build price view with open prices and daily change
    price_data = {}
    # Extend range by 1 day backward for pct_change on first backtest day
    price_start = prices_full.index[prices_full.index <= pd.Timestamp(actual_start_str)]
    price_start = price_start[-1] if len(price_start) > 0 else actual_start_str
    for name, code in etfs.items():
        close_full = prices_full[name].loc[price_start:end_str]
        close_show = close_full.loc[actual_start_str:end_str]
        price_data[f"{name} 收盘"] = close_show.round(3)
        if open_full is not None and name in open_full.columns:
            open_show = open_full[name].loc[actual_start_str:end_str]
            price_data[f"{name} 开盘"] = open_show.round(3)
            chg = close_full.pct_change().loc[actual_start_str:end_str]
            price_data[f"{name} 涨跌%"] = chg.apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        else:
            price_data[f"{name} 开盘"] = "—"
            price_data[f"{name} 涨跌%"] = "—"

    price_view = pd.DataFrame(price_data)
    price_view = price_view.sort_index(ascending=False)
    price_view.index = price_view.index.strftime("%Y-%m-%d")
    price_view.index.name = "日期"
    st.dataframe(price_view, height=400, width='stretch')
    st.caption("收盘价/开盘价（未填充），按日期倒序。NaN = 当日无交易或数据缺失。")

    if daily_signals:
        # ── 4. 每日信号 ──
        st.divider()
        st.markdown("## 📡 每日信号")
        strategy_labels_short = {
            "momentum": "动量轮动", "rsi": "RSI均值回归",
            "bb": "布林带均值回归", "macd": "MACD趋势跟随",
            "mom_rsi": "动量+RSI过滤", "mom_bb": "动量+布林带过滤",
            "vol_weighted": "波动率加权", "stop_loss": "动量+移动止损",
            "dual_lookback": "双周期动量", "trend_strength": "趋势确认动量",
        }
        st.markdown(f"### 📡 信号明细 — {strategy_labels_short.get(strategy, strategy)}")

        # Build trade date set for marking rows
        trade_date_set = {str(d)[:10] for d in trade_dates
                         if d >= pd.Timestamp(actual_start_str) and d <= pd.Timestamp(end_str)}

        sig_by_date = {}
        for s in daily_signals:
            s_dt = s.get('_dt')
            if s_dt is not None:
                dk = str(s_dt)[:10]
                sig_by_date[dk] = s

        # Show all signal dates in reverse order, latest first
        sig_dates = sorted(sig_by_date.keys())
        sig_dates = [d for d in sig_dates if actual_start_str <= d <= end_str]
        max_rows = 30
        sig_rows = []
        
        # Build trade markers, buy/sell prices
        signal_trade_dates = set()
        exec_buy_price = {}          # exec_date_str -> (etf_name, price_str)
        exec_sell_price = {}         # exec_date_str -> (etf_name, price_str)

        for tdt, told, tnew in trade_details:
            tdt_ts = pd.Timestamp(tdt) if not isinstance(tdt, pd.Timestamp) else tdt
            exec_dk = str(tdt_ts)[:10]
            signal_trade_dates.add(exec_dk)
            if exec_timing in ("T+1开盘", "T日开盘") and open_full is not None:
                if tnew and tnew in open_full.columns and tdt_ts in open_full.index:
                    exec_buy_price[exec_dk] = (tnew, f"{open_full[tnew].loc[tdt_ts]:.3f}")
                if told and told in open_full.columns and tdt_ts in open_full.index:
                    exec_sell_price[exec_dk] = (told, f"{open_full[told].loc[tdt_ts]:.3f}")
            else:
                if tnew and tdt_ts in prices_full.index and tnew in prices_full.columns:
                    exec_buy_price[exec_dk] = (tnew, f"{prices_full[tnew].loc[tdt_ts]:.3f}")
                if told and tdt_ts in prices_full.index and told in prices_full.columns:
                    exec_sell_price[exec_dk] = (told, f"{prices_full[told].loc[tdt_ts]:.3f}")

        # Initial position
        first_dates = [d for d in sorted(sig_by_date.keys()) if actual_start_str <= d <= end_str]
        if first_dates:
            fd = first_dates[0]
            fh = sig_by_date[fd].get('holding') or 'CASH'
            if fh != 'CASH' and fd not in signal_trade_dates:
                signal_trade_dates.add(fd)
                fdt = pd.Timestamp(fd)
                if exec_timing == "T日开盘":
                    if fdt in open_full.index and fh in open_full.columns:
                        exec_buy_price[fd] = (fh, f"{open_full[fh].loc[fdt]:.3f}")
                elif exec_timing == "T+1开盘":
                    next_d = prices_full.index[prices_full.index > fdt]
                    if len(next_d) > 0 and fh in open_full.columns and next_d[0] in open_full.index:
                        exec_buy_price[str(next_d[0])[:10]] = (fh, f"{open_full[fh].loc[next_d[0]]:.3f}")
                else:
                    if fdt in prices_full.index and fh in prices_full.columns:
                        exec_buy_price[fd] = (fh, f"{prices_full[fh].loc[fdt]:.3f}")

        sig_rows = []
        for dk in sig_dates[-max_rows:]:
            match = sig_by_date[dk]; td = pd.Timestamp(dk)
            holding = match.get('holding', 'CASH') or 'CASH'
            is_trade = dk in signal_trade_dates
            hlabel = f"{holding} ({etf_codes.get(holding, '')})" if holding != 'CASH' else 'CASH'
            buy_info = exec_buy_price.get(dk)
            sell_info = exec_sell_price.get(dk)
            buy_px_str = "—"
            sell_px_str = "—"
            if buy_info and not (exec_timing == "T+1开盘" and is_trade):
                buy_px_str = buy_info[1]
            if sell_info:
                sell_px_str = sell_info[1]

            # 今日信号：基于当日数据的理想持仓（与实持可能不同，如MOO T+1执行或非检查日）
            best_today = None
            best_qualifiers = {}
            for name, code in etfs.items():
                val = match.get(name)
                if val is not None:
                    px_val = prices_full[name].get(td, np.nan) if td in prices_full.index else np.nan
                    if pd.notna(px_val):
                        p_ffill = prices_full[name].ffill()
                        ma_val = p_ffill.rolling(ma_days).mean().get(td, np.nan)
                        if pd.notna(ma_val) and px_val > ma_val:
                            best_qualifiers[name] = val
            best_today = max(best_qualifiers, key=best_qualifiers.get) if best_qualifiers else None
            best_label = f"{best_today} ({etf_codes.get(best_today, '')})" if best_today else "CASH"
            suggest_changed = (best_today != holding)

            row = {"日期": dk, "持仓": hlabel,
                   "今日信号": f"→ {best_label}" if suggest_changed else best_label,
                   "调仓": "🔄" if is_trade else "",
                   "买入价格": buy_px_str,
                   "卖出价格": sell_px_str}
            # Track which ETF cols to highlight for this row
            buy_etf = buy_info[0] if buy_info else None
            sell_etf = sell_info[0] if sell_info else None

            for name, code in etfs.items():
                val = match.get(name)
                open_px = "—"; close_px = "—"; chg_str = "—"
                if open_full is not None and name in open_full.columns and td in open_full.index:
                    o = open_full[name].loc[td]; open_px = f"{o:.3f}" if pd.notna(o) else "—"
                if td in prices_full.index and name in prices_full.columns:
                    px_today = prices_full[name].loc[td]
                    if pd.notna(px_today): close_px = f"{px_today:.3f}"
                    prev_idx = prices_full.index[prices_full.index < td]
                    if len(prev_idx) > 0:
                        px_prev = prices_full[name].loc[prev_idx[-1]]
                        if pd.notna(px_today) and pd.notna(px_prev) and px_prev > 0:
                            chg_str = f"{(px_today / px_prev - 1):+.3%}"
                if val is not None:
                    indicator_str = f"{round(val, 4)}"
                    px_val = prices_full[name].get(td, np.nan) if td in prices_full.index else np.nan
                    if pd.notna(px_val):
                        p_ffill = prices_full[name].ffill()
                        ma_val = p_ffill.rolling(ma_days).mean().get(td, np.nan)
                        if pd.notna(ma_val) and px_val <= ma_val: indicator_str += " ✗MA"
                    else: indicator_str += " ✗数据"
                    row[f"{name} 指标"] = indicator_str
                else: row[f"{name} 指标"] = "—"
                # Highlight buy/sell on execution price column
                if exec_timing in ("T+1开盘", "T日开盘"):
                    op = open_px if open_px != "—" else "—"
                    row[f"{name} 开盘"] = ("▶" if name==buy_etf else "") + ("◀" if name==sell_etf else "") + (f" {op}" if op!="—" else "—")
                    row[f"{name} 收盘"] = close_px
                else:
                    cp = close_px if close_px != "—" else "—"
                    row[f"{name} 收盘"] = ("▶" if name==buy_etf else "") + ("◀" if name==sell_etf else "") + (f" {cp}" if cp!="—" else "—")
                    row[f"{name} 开盘"] = open_px
                row[f"{name} 涨幅"] = chg_str
            sig_rows.append(row)
        sig_rows.reverse()  # show latest first
        if sig_rows:
            n_trade = sum(1 for r in sig_rows if r["调仓"] == "🔄")
            n_diff = sum(1 for r in sig_rows if "→" in str(r.get("今日信号", "")))
            st.caption(
                f"显示最近 {len(sig_rows)} 个信号日（日期倒序）。"
                f"其中 {n_trade} 天调仓（🔄），{n_diff} 天持仓≠信号。"
                f"「今日信号」= 当日理想的持仓（MOO下T+1执行/非检查日均可能与实持不同）。"
                f"数值为策略排名指标（动量=ROC，RSI=RSI值，BB=%B，MACD=柱状线），越大越优先。"
                f"「—」= 当日数据缺失。「✗MA」= 指标有效但未通过MA{ma_days}趋势过滤。「✗数据」= 当日无交易数据。"
                f" ▶🟢=买入 ◀🔴=卖出"
            )
            df_sig = pd.DataFrame(sig_rows)
            # Color buy/sell price cells: green for buy, red for sell
            def _highlight_prices(val):
                if isinstance(val, str) and val.startswith("▶"):
                    return 'background-color: #d4edda; color: #155724'  # green
                elif isinstance(val, str) and val.startswith("◀"):
                    return 'background-color: #f8d7da; color: #721c24'  # red
                return ''
            styled = df_sig.style.map(_highlight_prices)
            st.dataframe(styled, height=400, hide_index=True, width='stretch')

    # ── 5. 全部组合对比 ──
    if compare_all:
        st.divider()
        st.markdown("## 📋 全部组合对比")
        st.subheader("📊 全部组合对比", divider="blue")

        all_groups = dict(cfg["groups"])
        compare_rows = []

        # Load all unique ETF codes once (not per-group), then slice in-memory.
        # Use unique keys (name__code) since different groups may use different codes
        # for the same category name (e.g., "红利低波"→512890 vs "红利低波"→515080).
        all_unique: dict[str, str] = {}
        for getfs in all_groups.values():
            for name, code in getfs.items():
                all_unique[f"{name}__{code}"] = code
        all_prices = cached_prices(all_unique, "_all", source=source)
        all_prices = all_prices[all_prices.index >= lookback]
        all_open = cached_open_prices(all_unique, "_all", source=source) if use_backtrader else None
        if all_open is not None:
            all_open = all_open[all_open.index >= lookback]
        all_midday = load_midday_prices(all_unique) if _use_midday else None
        all_aft_open = load_afternoon_open_prices(all_unique) if _use_midday else None

        group_data = {}
        for gname, getfs in all_groups.items():
            # Map unique column names back to group's category names
            col_map = {f"{name}__{code}": name for name, code in getfs.items()}
            gprices_full = all_prices[list(col_map.keys())].rename(columns=col_map).dropna(how="all")
            if len(gprices_full) == 0:
                continue
            gopen_full = all_open[list(col_map.keys())].rename(columns=col_map) if all_open is not None else None
            gmidday = all_midday[list(col_map.keys())].rename(columns=col_map) if all_midday is not None else None
            gaft_open = all_aft_open[list(col_map.keys())].rename(columns=col_map) if all_aft_open is not None else None

            gfirst = {name: gprices_full[name].first_valid_index() for name in getfs}
            valid_firsts = [d for d in gfirst.values() if d is not None]
            glatest = max(valid_firsts) if valid_firsts else pd.Timestamp(start_str)
            gwarmup = gprices_full.index[min(ma_days, len(gprices_full) - 1)]
            gactual = max(glatest, gwarmup, pd.Timestamp(start_str))
            gactual_str = gactual.strftime("%Y-%m-%d")

            group_data[gname] = (gprices_full, gopen_full, gactual_str, gmidday, gaft_open)

        def _run_one_group(gname, gprices_full, gopen_full, gactual_str, gmidday=None, gaft_open=None):
            """Run backtest for one group, returns list of row dicts."""
            rows = []
            for m in modes_to_run:
                if use_backtrader:
                    bt_m = 'moc' if gopen_full is None else 'moo'
                    gnav, gbnav, gret, gbret, gtrades, gtd, gtdets, _ = \
                        run_backtest_bt(gprices_full, m, gactual_str, end_str, ma_days, roc_days,
                                        strategy=bt_m, open_prices=gopen_full,
                                        exec_mode=bt_m, delay=delay)
                else:
                    gnav, gbnav, gret, gbret, gtrades, gtd, gtdets, _ = \
                        run_backtest(gprices_full, m, gactual_str, end_str, ma_days, roc_days,
                                     open_prices=gopen_full,
                                     midday_prices=gmidday,
                                     afternoon_open_prices=gaft_open,
                                     delay=delay, use_open_signal=_use_open_signal)
                gm = calc_metrics(gnav, gret)
                gwr = trade_win_rate(gret, gtdets, gprices_full)
                rows.append({
                    "组合": gname, "模式": m.upper(),
                    "累计收益": f"{gm.get('累计收益', 0):.3%}",
                    "年化收益": f"{gm.get('年化收益', 0):.3%}",
                    "夏普比率": f"{gm.get('夏普比率', 0):.2f}",
                    "最大回撤": f"{gm.get('最大回撤', 0):.3%}",
                    "卡尔玛比率": f"{gm.get('卡尔玛比率', 0):.2f}",
                    "胜率": f"{gwr:.0%}", "交易次数": gtrades,
                })
            return rows

        total_groups = len(group_data)
        prog_bar = st.progress(0, text=f"0/{total_groups} 组合...")
        done = 0
        with ThreadPoolExecutor(max_workers=min(4, len(all_groups))) as executor:
            futures = {executor.submit(_run_one_group, gname, *gdata): gname
                       for gname, gdata in group_data.items()}
            for future in as_completed(futures):
                gname = futures[future]
                try:
                    compare_rows.extend(future.result())
                except Exception as e:
                    import traceback
                    st.error(f"**{gname}** 回测失败:\n```\n{traceback.format_exc()}\n```")
                    compare_rows.append({
                        "组合": gname, "模式": "—",
                        "累计收益": f"❌ {e}", "年化收益": 0.0, "夏普比率": 0.0,
                        "最大回撤": 0.0, "卡尔玛比率": 0.0, "胜率": 0.0, "交易次数": 0,
                    })
                done += 1
                prog_bar.progress(done / total_groups, text=f"{done}/{total_groups} 组合 — {gname} 完成")
        prog_bar.empty()

        if compare_rows:
            df_cmp = pd.DataFrame(compare_rows)
            st.dataframe(df_cmp, hide_index=True, width='stretch')

    # ── 6. 参数遍历 ──
    if optimize:
        st.divider()
        st.markdown("## 🔍 参数遍历")
        st.subheader("🔍 参数遍历结果", divider="orange")

        ma_range = list(range(10, 201, opt_ma_step))
        roc_range = list(range(5, 121, opt_roc_step))
        total_combo = len(ma_range) * len(roc_range) * len(modes_to_run)

        with st.status(f"搜索 {len(ma_range)}×{len(roc_range)} = {total_combo} 种组合...", expanded=False) as status:
            prog = st.progress(0, text="初始化...")
            df_opt = grid_search(prices_full, modes_to_run, actual_start_str, end_str,
                                 ma_range, roc_range, prog,
                                 open_prices=_exec_open,
                                 midday_prices=_midday_prices,
                                 afternoon_open_prices=_afternoon_open_prices,
                                 delay=delay, use_open_signal=_use_open_signal)
            status.update(label=f"完成 {total_combo} 种组合", state="complete")

        # ── Result table per mode ──
        opt_tab = st.tabs([f"{m.upper()}调仓 TOP{opt_top}" for m in modes_to_run] + ["全部 TOP"])
        for ti, mode in enumerate(modes_to_run):
            with opt_tab[ti]:
                sub = df_opt[df_opt["模式"] == mode].sort_values(opt_metric, ascending=False).head(opt_top).copy()
                _fmt_optimize_table(sub)
                st.dataframe(sub, hide_index=True)

                # Heatmap for this mode
                figs = build_optimizer_heatmap(df_opt, opt_metric)
                if mode in figs:
                    st.plotly_chart(figs[mode], width='stretch')

        # ── All-mode summary ──
        with opt_tab[-1]:
            top_all = df_opt.sort_values(opt_metric, ascending=False).head(opt_top).copy()
            _fmt_optimize_table(top_all)
            st.dataframe(top_all, hide_index=True)

        # CSV download
        csv = df_opt.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 下载全量CSV", csv, "etf_optimize.csv", "text/csv", width='stretch')

def _strategy_signal_for_date(prices, target_date, strategy, ma_days=60, roc_days=20,
                              open_prices=None):
    """计算指定策略在指定日期的信号和指标值"""
    dt = pd.Timestamp(target_date)
    if dt not in prices.index:
        available = prices.index[prices.index <= dt]
        if len(available) == 0:
            return None, None, None
        dt = available[-1]

    # Find previous valid date for daily change calculation
    prev_dt = prices.index[prices.index < dt]
    prev_dt = prev_dt[-1] if len(prev_dt) > 0 else None

    p = prices.ffill()
    rows = []
    candidates = {}

    for name in prices.columns:
        px = float(prices[name].loc[dt]) if dt in prices.index else float('nan')
        is_valid = dt in prices[name].dropna().index

        row = {"ETF": name}

        # Open price
        if open_prices is not None and name in open_prices.columns and dt in open_prices.index:
            row["开盘价"] = float(open_prices[name].loc[dt])
        else:
            row["开盘价"] = None

        row["收盘价"] = px

        # Daily change
        if prev_dt is not None and prev_dt in prices.index:
            prev_px = prices[name].loc[prev_dt]
            if not pd.isna(px) and not pd.isna(prev_px) and prev_px > 0:
                row["涨幅"] = (px / prev_px) - 1
            else:
                row["涨幅"] = None
        else:
            row["涨幅"] = None

        if strategy == "momentum":
            ma = float(p[name].rolling(ma_days).mean().loc[dt])
            roc = float(p[name].pct_change(roc_days, fill_method=None).loc[dt])
            row[f"MA{ma_days}"] = ma
            row[f"ROC{roc_days}"] = roc
            if is_valid and not pd.isna(ma) and px > ma and not pd.isna(roc):
                candidates[name] = roc
        elif strategy == "rsi":
            delta = p[name].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.loc[dt])
            row["RSI(14)"] = rsi_val
            if is_valid and not pd.isna(rsi_val) and rsi_val < 40:
                candidates[name] = rsi_val
        elif strategy == "bb":
            sma = p[name].rolling(20).mean()
            std = p[name].rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            pct_b = (px - float(lower.loc[dt])) / (float(upper.loc[dt]) - float(lower.loc[dt])) if float(upper.loc[dt]) != float(lower.loc[dt]) else float('nan')
            row["BB(20,2) %B"] = pct_b
            if is_valid and not pd.isna(pct_b) and pct_b < 0.3:
                candidates[name] = pct_b
        elif strategy == "macd":
            ema12 = p[name].ewm(span=12).mean()
            ema26 = p[name].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = float(macd_line.loc[dt]) - float(signal_line.loc[dt])
            row["MACD柱状线"] = histogram
            if is_valid and not pd.isna(histogram) and histogram > 0:
                candidates[name] = histogram
        elif strategy == "mom_rsi":
            ma = float(p[name].rolling(ma_days).mean().loc[dt])
            roc = float(p[name].pct_change(roc_days, fill_method=None).loc[dt])
            delta = p[name].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.loc[dt])
            row[f"MA{ma_days}"] = ma
            row[f"ROC{roc_days}"] = roc
            row["RSI(14)"] = rsi_val
            if is_valid and not pd.isna(ma) and px > ma and not pd.isna(roc) and not pd.isna(rsi_val) and rsi_val < 70:
                candidates[name] = roc
        elif strategy == "mom_bb":
            ma = float(p[name].rolling(ma_days).mean().loc[dt])
            roc = float(p[name].pct_change(roc_days, fill_method=None).loc[dt])
            sma = p[name].rolling(20).mean()
            std = p[name].rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            pct_b = (px - float(lower.loc[dt])) / (float(upper.loc[dt]) - float(lower.loc[dt])) if float(upper.loc[dt]) != float(lower.loc[dt]) else float('nan')
            row[f"MA{ma_days}"] = ma
            row[f"ROC{roc_days}"] = roc
            row["BB(20,2) %B"] = pct_b
            if is_valid and not pd.isna(ma) and px > ma and not pd.isna(roc) and not pd.isna(pct_b) and pct_b < 0.8:
                candidates[name] = roc
        elif strategy == "vol_weighted":
            ma = float(p[name].rolling(ma_days).mean().loc[dt])
            roc = float(p[name].pct_change(roc_days, fill_method=None).loc[dt])
            vol = float(p[name].pct_change(fill_method=None).rolling(60).std().loc[dt])
            sharpe = roc / (vol * math.sqrt(252)) if not pd.isna(vol) and vol > 0 else float('nan')
            row[f"MA{ma_days}"] = ma
            row[f"ROC{roc_days}"] = roc
            row["年化Sharpe"] = sharpe
            if is_valid and not pd.isna(ma) and px > ma and not pd.isna(roc) and not pd.isna(sharpe):
                candidates[name] = sharpe
        elif strategy == "stop_loss":
            ma = float(p[name].rolling(ma_days).mean().loc[dt])
            roc = float(p[name].pct_change(roc_days, fill_method=None).loc[dt])
            row[f"MA{ma_days}"] = ma
            row[f"ROC{roc_days}"] = roc
            if is_valid and not pd.isna(ma) and px > ma and not pd.isna(roc):
                candidates[name] = roc

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add MA pass column
    ma_col = f"MA{ma_days}"
    if ma_col in df.columns:
        df["MA通过"] = df.apply(
            lambda r: "✓" if (pd.notna(r.get(ma_col)) and pd.notna(r["收盘价"])
                              and r["收盘价"] > r[ma_col]) else "✗", axis=1)
    else:
        df["MA通过"] = "—"

    # Add ranking column based on strategy's scoring metric
    rank_cfg = {
        "momentum": (f"ROC{roc_days}", "desc"), "rsi": ("RSI(14)", "asc"),
        "bb": ("BB(20,2) %B", "asc"), "macd": ("MACD柱状线", "desc"),
        "mom_rsi": (f"ROC{roc_days}", "desc"), "mom_bb": (f"ROC{roc_days}", "desc"),
        "vol_weighted": (f"ROC{roc_days}", "desc"), "stop_loss": (f"ROC{roc_days}", "desc"),
        "dual_lookback": (f"ROC{roc_days}", "desc"), "trend_strength": (f"ROC{roc_days}", "desc"),
    }
    rc = rank_cfg.get(strategy)
    if rc:
        rank_col, sort_order = rc
        if rank_col in df.columns:
            ascending = sort_order == "asc"
            df["排名"] = df[rank_col].rank(ascending=ascending, na_option="bottom").astype("Int64")

    if strategy in ("momentum", "mom_rsi", "mom_bb", "macd", "stop_loss"):
        best = max(candidates, key=candidates.get) if candidates else None
    elif strategy == "vol_weighted":
        best = max(candidates, key=candidates.get) if candidates else None
    elif strategy in ("rsi", "bb"):
        best = min(candidates, key=candidates.get) if candidates else None
    else:
        best = None

    return best, df, dt


# ── Signal query ─────────────────────────────────────────
if sig_btn:
    etfs = cfg["groups"][sel_group]
    strategy_labels = {"momentum": "动量轮动", "rsi": "RSI均值回归", "bb": "布林带均值回归", "macd": "MACD趋势跟随", "mom_rsi": "动量+RSI过滤", "mom_bb": "动量+布林带过滤", "vol_weighted": "波动率加权", "dual_lookback": "双周期动量", "trend_strength": "趋势确认动量", "stop_loss": "动量+移动止损"}
    with st.spinner("查询信号..."):
        prices = cached_prices(etfs, sel_group, source=source)
        open_prices = cached_open_prices(etfs, sel_group, source=source)
        target_dt = pd.Timestamp(sig_date.strftime("%Y-%m-%d"))
        last_data_date = prices.index[-1]

        # 数据新鲜度检查
        missing_etfs = []
        stale_etfs = {}
        today = pd.Timestamp.now().normalize()
        if last_data_date < today - pd.Timedelta(days=1):
            st.warning(f"⚠️ 缓存数据最新日期为 {last_data_date.strftime('%Y-%m-%d')}，可能不是最新。如需拉取最新数据请点「刷新数据缓存」按钮。")
        if target_dt > last_data_date:
            st.warning(f"⚠️ 查询日期 {target_dt.strftime('%Y-%m-%d')} 超出数据范围，使用最新数据 {last_data_date.strftime('%Y-%m-%d')} 计算信号")
            sig_date_actual = last_data_date
        else:
            sig_date_actual = target_dt
            for name in etfs:
                if target_dt in prices.index:
                    px = prices[name].loc[target_dt]
                    if pd.isna(px):
                        # 找到最近的可用日期
                        valid = prices[name].loc[:target_dt].dropna()
                        if len(valid) > 0:
                            stale_etfs[name] = valid.index[-1]
                        else:
                            missing_etfs.append(name)

        best, df, actual_dt = _strategy_signal_for_date(
            prices, sig_date_actual.strftime("%Y-%m-%d"), strategy, ma_days, roc_days,
            open_prices=open_prices)

    if df is None:
        st.warning("数据不足，无法查询")
    else:
        st.subheader(f"信号 [{strategy_labels.get(strategy, strategy)}]: {actual_dt.strftime('%Y-%m-%d')} [{sel_group}]")

        # 数据新鲜度警告
        if stale_etfs:
            stale_list = "、".join(f"{name}(最近: {d.strftime('%m-%d')})" for name, d in stale_etfs.items())
            st.warning(f"⚠️ 以下ETF在查询日期无数据，使用了前值填充(ffill)：{stale_list}。信号可能基于非真实数据，请谨慎参考。")

        # Check for systematic data gaps (e.g. source partially failing)
        recent = prices.iloc[-10:]
        gap_etfs = []
        for name in prices.columns:
            if recent[name].isna().all():
                last_valid = prices[name].dropna().index[-1]
                gap_etfs.append((name, last_valid))
        if gap_etfs:
            gap_list = "、".join(f"{n}(最近: {d.strftime('%m-%d')})" for n, d in gap_etfs)
            st.warning(f"⚠️ 以下ETF近期完全无数据（数据源可能部分失效）：{gap_list}。建议切换数据源。")
        if missing_etfs:
            missing_list = "、".join(missing_etfs)
            st.error(f"❌ 以下ETF完全没有可用数据：{missing_list}")

        # Add ETF codes to display
        etf_codes_map = {name: code for name, code in etfs.items()}
        if best:
            best_code = etf_codes_map.get(best, '')
            st.success(f"持有 **{best} ({best_code})**")
        else:
            st.warning("空仓")

        df["ETF"] = df["ETF"].apply(lambda n: f"{n} ({etf_codes_map.get(n, '')})")
        # Format 涨幅 as percentage
        if "涨幅" in df.columns:
            df["涨幅"] = df["涨幅"].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        if "开盘价" in df.columns:
            df["开盘价"] = df["开盘价"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        if "收盘价" in df.columns:
            df["收盘价"] = df["收盘价"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(df, hide_index=True, width='content')
