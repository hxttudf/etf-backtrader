"""ETF双动量轮动 — backtrader 引擎

提供与原手写回测完全相同的接口，通过 backtrader 执行回测逻辑。
信号逻辑与原引擎 100% 一致：
- 信号条件使用 raw 价格 (NaN → 不满足 px > ma 条件)
- 指标 (MA / ROC) 使用 ffill 价格计算
- T+1 执行：信号在 bar i 收盘后计算，bar i+1 open 执行
"""

import math
import sys as _sys
from datetime import datetime

import numpy as np
import pandas as pd

import backtrader as bt

from etf_data import calc_indicators

# Lazy ref to manual run_backtest (lives in etf_app / __main__ when run as Streamlit)
_manual_run_backtest = None


def _get_manual_bt():
    global _manual_run_backtest
    if _manual_run_backtest is None:
        for _mod_name in ('__main__', 'etf_app'):
            _m = _sys.modules.get(_mod_name)
            if _m and hasattr(_m, 'run_backtest'):
                _manual_run_backtest = _m.run_backtest
                break
    return _manual_run_backtest


class StampDutyCommission(bt.CommInfoBase):
    """A股ETF佣金: 万1双向 + 万5印花税(仅卖出)"""
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
        ('commission', 0.0001),
        ('stamp_duty', 0.0005),
    )

    def _getcommission(self, size, price, pseudoexec):
        comm = abs(size) * price * self.p.commission
        if size < 0:
            comm += abs(size) * price * self.p.stamp_duty
        return comm


def _make_data_feed(prices, name, open_prices=None, ma_df=None, roc_df=None):
    """创建 backtrader 数据源。

    - close/MA/ROC 来自手动引擎 calc_indicators，保证信号完全一致
    - openinterest: 0 = 当日原始数据为 NaN，1 = 有效
    """
    raw = prices[name]
    raw = raw[~raw.index.duplicated(keep='last')]
    is_valid = (~raw.isna()).astype(float)
    close_filled = raw.ffill()

    close_arr = close_filled.values
    if open_prices is not None and name in open_prices.columns:
        open_raw = open_prices[name]
        open_raw = open_raw[~open_raw.index.duplicated(keep='last')]
        open_aligned = open_raw.reindex(close_filled.index)
        fallback_open = pd.Series(np.roll(close_arr, 1), index=close_filled.index)
        fallback_open.iloc[0] = close_arr[0]
        open_aligned = open_aligned.fillna(fallback_open)
        open_arr = open_aligned.values
    else:
        open_arr = np.roll(close_arr, 1)
        open_arr[0] = close_arr[0]

    df = pd.DataFrame(index=close_filled.index)
    df['close'] = close_arr
    df['open'] = open_arr
    df['high'] = df[['open', 'close']].max(axis=1)
    df['low'] = df[['open', 'close']].min(axis=1)
    df['volume'] = 100
    df['openinterest'] = is_valid.values
    # Inject pre-computed MA/ROC as extra lines
    if ma_df is not None and name in ma_df.columns:
        ma_aligned = ma_df[name].reindex(close_filled.index).fillna(np.nan).values
        df['ma_pre'] = ma_aligned
    if roc_df is not None and name in roc_df.columns:
        roc_aligned = roc_df[name].reindex(close_filled.index).fillna(np.nan).values
        df['roc_pre'] = roc_aligned

    # Custom PandasData to expose pre-computed MA/ROC (cols 6,7 in df)
    class _ETFData(bt.feeds.PandasData):
        lines = ('ma_pre', 'roc_pre',)
        params = (('ma_pre', 6), ('roc_pre', 7),)
    data = _ETFData(dataname=df)
    data._name = name
    return data


# ── Helper: trade execution (shared by all strategies) ──────────────

def _execute_trade(strat, dt, new_holding):
    """Execute a position change. With coc/coo, orders fill at same bar (close or open)."""
    if strat._holding is not None:
        for d in strat.datas:
            if d._name == strat._holding:
                strat.order_target_percent(d, target=0.0)
                break
    if new_holding is not None:
        for d in strat.datas:
            if d._name == new_holding:
                strat.order_target_percent(d, target=0.999)
                break
    strat._trade_log.append((dt, strat._holding, new_holding))
    strat._holding = new_holding
    strat._last_trade_bar = len(strat)


# ── Strategies ──────────────────────────────────────────────────────

class MomentumRotation(bt.Strategy):
    """双动量轮动: MA60趋势过滤 + ROC20动量排名，持有最优单一ETF

    信号判定与原手写引擎完全一致：
    - MA/ROC 基于 ffill 后的收盘价计算
    - 仅当原始收盘价有效 (openinterest > 0) 且 px > ma 时才纳入候选
    - T+1 执行：信号在 bar i 收盘计算，backtrader 在 bar i+1 open 执行
    """
    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
            }

        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            px = d.close[0]
            if not np.isnan(ma_val) and px > ma_val and not np.isnan(roc_val):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None

        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            roc_val = self.inds[name]['roc'][0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class MOCRotation(bt.Strategy):
    """MOC 收盘执行: coc=True, 信号 close[-1] → 执行 close[0]

    - 信号源: close[-1] = T-1日收盘 (与 MOO 一致)
    - 执行: broker.set_coc(True) → 订单在同 bar close[0] 成交
    - sell old + buy new 同价同时执行, 无资金错位
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 20),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
        ('start_date', None),
        ('end_date', None),
        ('ma_df', None),
        ('roc_df', None),
    )

    def __init__(self):
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999
        self._ma = self.p.ma_df
        self._roc = self.p.roc_df

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        dt_ts = pd.Timestamp(dt)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            px = d.close[0]
            # Look up pre-computed MA/ROC by date (correct alignment)
            ma_val = self._ma[name].get(dt_ts, np.nan) if self._ma is not None and name in self._ma.columns else np.nan
            roc_val = self._roc[name].get(dt_ts, np.nan) if self._roc is not None and name in self._roc.columns else np.nan
            if not np.isnan(ma_val) and px > ma_val and not np.isnan(roc_val):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None

        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            roc_val = self._roc[name].get(dt_ts, np.nan) if self._roc is not None and name in self._roc.columns else np.nan
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class MOORotation(bt.Strategy):
    """MOO 开盘执行: coo=True, next() 算信号, next_open() 执行

    - 信号源: close[0] in next() = close[i] (当日收盘)
    - next_open() 在下一 bar 用存储信号执行, coo 在 open[0] 成交
    - sell old + buy new 同价同时执行, 无资金错位
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 20),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
        ('start_date', None),
        ('end_date', None),
        ('ma_df', None),
        ('roc_df', None),
    )

    def __init__(self):
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999
        self._pending_signal = None
        self._started_in_range = False
        self._ma = self.p.ma_df
        self._roc = self.p.roc_df

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        # Clean start: reset holding on first bar in backtest range
        if self.p.start_date and not self._started_in_range:
            sd = pd.Timestamp(self.p.start_date)
            ed = pd.Timestamp(self.p.end_date) if self.p.end_date else pd.Timestamp.now()
            if sd <= dt <= ed:
                self._holding = None
                self._started_in_range = True

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if should_check and len(self) >= self.p.ma_days:
            above = {}
            for d in self.datas:
                name = d._name
                if d.openinterest[0] == 0:
                    continue
                i = len(self) - 1
                ma_val = self._ma[name].iloc[i] if self._ma is not None and name in self._ma.columns else np.nan
                roc_val = self._roc[name].iloc[i] if self._roc is not None and name in self._roc.columns else np.nan
                px = d.close[0]
                if not np.isnan(ma_val) and px > ma_val and not np.isnan(roc_val):
                    above[name] = roc_val
            self._pending_signal = max(above, key=above.get) if above else None
        else:
            self._pending_signal = None

        sig_record = {'_dt': dt}
        i = len(self) - 1
        for d in self.datas:
            name = d._name
            roc_val = self._roc[name].iloc[i] if self._roc is not None and name in self._roc.columns else np.nan
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
        sig_record['holding'] = self._pending_signal
        self._daily_signals.append(sig_record)

    def next_open(self):
        dt = self.datas[0].datetime.datetime(0)
        # Clean start: skip first bar's stale signal
        if self.p.start_date and not self._started_in_range:
            sd = pd.Timestamp(self.p.start_date)
            ed = pd.Timestamp(self.p.end_date) if self.p.end_date else pd.Timestamp.now()
            if sd <= dt <= ed:
                self._holding = None
                self._pending_signal = None
                self._started_in_range = True
                return
        if self._pending_signal != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold \
                and self._pending_signal is not None:
            _execute_trade(self, dt, self._pending_signal)


class StopLossMomentum(bt.Strategy):
    """动量轮动 + 移动止损: 从持仓高点的最大回撤超过阈值时空仓

    信号逻辑与 MomentumRotation 完全一致，但增加止损层:
    入场后记录峰值价格，价格从峰值回落超过 stop_loss_pct 时空仓。
    T+1 执行：信号/止损在 bar i 计算，backtrader 在 bar i+1 open 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('stop_loss_pct', 0.05),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999
        self._peak_price = 0.0

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        # Stop loss check: daily, on current holding
        stop_loss_triggered = False
        if self._holding is not None:
            for d in self.datas:
                if d._name == self._holding and d.openinterest[0] > 0:
                    px = d.close[0]
                    self._peak_price = max(self._peak_price, px)
                    if px < self._peak_price * (1 - self.p.stop_loss_pct):
                        self.close(data=d)
                        self._trade_log.append((dt, self._holding, None))
                        self._holding = None
                        self._last_trade_bar = len(self)
                        self._peak_price = 0.0
                        stop_loss_triggered = True
                    break

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            px = d.close[0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
            if not np.isnan(ma_val) and px > ma_val and not np.isnan(roc_val):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if not stop_loss_triggered and new_holding is not None \
                and new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)

        # Reset peak on position change
        if new_holding != self._holding:
            self._peak_price = 0.0


class RSIMeanReversion(bt.Strategy):
    """RSI均值回归轮动: 买入最超卖的ETF，持有单一最优ETF

    RSI(14) < oversold_threshold 才纳入候选，选 RSI 最低的 ETF。
    无候选时空仓。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('rsi_period', 14),
        ('oversold_threshold', 40),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'rsi': bt.indicators.RSI(d.close, period=self.p.rsi_period),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        candidates = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            rsi_val = self.inds[name]['rsi'][0]
            if not np.isnan(rsi_val):
                sig_record[name] = float(rsi_val)
                if rsi_val < self.p.oversold_threshold:
                    candidates[name] = rsi_val

        new_holding = min(candidates, key=candidates.get) if candidates else None
        sig_record['holding'] = new_holding
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class BollingerBandsRotation(bt.Strategy):
    """布林带均值回归轮动: 买入最接近下轨的ETF，持有单一最优ETF

    BB(20,2) %B < 0.3 才纳入候选，选 %B 最低的 ETF。
    %B = (close - lower) / (upper - lower)。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('entry_threshold', 0.3),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'bb': bt.indicators.BollingerBands(
                    d.close, period=self.p.bb_period, devfactor=self.p.bb_dev
                ),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        candidates = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            bb = self.inds[name]['bb']
            top = bb.top[0]
            bot = bb.bot[0]
            if np.isnan(top) or np.isnan(bot) or top == bot:
                continue
            pct_b = (d.close[0] - bot) / (top - bot)
            sig_record[name] = float(pct_b)
            if pct_b < self.p.entry_threshold:
                candidates[name] = pct_b

        new_holding = min(candidates, key=candidates.get) if candidates else None
        sig_record['holding'] = new_holding
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class MACDTrendFollowing(bt.Strategy):
    """MACD趋势跟随轮动: 买入MACD柱状线最强的ETF，持有单一最优ETF

    MACD(12,26,9) 柱状线 > 0 才纳入候选，选柱状线最高的 ETF。
    柱状线 = MACD线 - 信号线。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'macd': bt.indicators.MACD(
                    d.close,
                    period_me1=self.p.macd_fast,
                    period_me2=self.p.macd_slow,
                    period_signal=self.p.macd_signal,
                ),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        candidates = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            macd = self.inds[name]['macd']
            macd_line = macd.macd[0]
            signal_line = macd.signal[0]
            if np.isnan(macd_line) or np.isnan(signal_line):
                continue
            histogram = macd_line - signal_line
            sig_record[name] = float(histogram)
            if histogram > 0:
                candidates[name] = histogram

        new_holding = max(candidates, key=candidates.get) if candidates else None
        sig_record['holding'] = new_holding
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class MomentumRSI(bt.Strategy):
    """动量轮动 + RSI 过滤: MA60趋势 → ROC25排名 → RSI防追高 → 选最优

    在动量轮动基础上，额外排除 RSI(14) > rsi_max 的 ETF，避免追高买入。
    T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('rsi_period', 14),
        ('rsi_max', 70),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
                'rsi': bt.indicators.RSI(d.close, period=self.p.rsi_period),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            rsi_val = self.inds[name]['rsi'][0]
            px = d.close[0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
            sig_record[f'rsi_{name}'] = float(rsi_val) if not np.isnan(rsi_val) else None
            if (not np.isnan(ma_val) and px > ma_val
                    and not np.isnan(roc_val) and not np.isnan(rsi_val)
                    and rsi_val < self.p.rsi_max):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class MomentumBB(bt.Strategy):
    """动量轮动 + 布林带过滤: MA60趋势 → ROC25排名 → BB防假突破 → 选最优

    在动量轮动基础上，额外排除 %B > bb_pct_max 的 ETF，避免假突破追高。
    T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('bb_pct_max', 0.8),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
                'bb': bt.indicators.BollingerBands(
                    d.close, period=self.p.bb_period, devfactor=self.p.bb_dev
                ),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            bb = self.inds[name]['bb']
            top = bb.top[0]
            bot = bb.bot[0]
            px = d.close[0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
            if np.isnan(top) or np.isnan(bot) or top == bot:
                continue
            pct_b = (px - bot) / (top - bot)
            sig_record[f'bb_{name}'] = float(pct_b)
            if (not np.isnan(ma_val) and px > ma_val
                    and not np.isnan(roc_val)
                    and pct_b < self.p.bb_pct_max):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class VolWeightedMomentum(bt.Strategy):
    """波动率加权动量: ROC/波动率排名，选单位风险收益最优的ETF

    与原始动量不同: 排名用的是年化Sharpe (ROC/波动率)，而不是裸ROC。
    这惩罚高波动ETF，奖励"稳稳上涨"的ETF。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('vol_days', 60),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
                'std': bt.indicators.StdDev(d.close, period=self.p.vol_days),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            std_val = self.inds[name]['std'][0]
            px = d.close[0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None
            if (not np.isnan(ma_val) and px > ma_val
                    and not np.isnan(roc_val) and not np.isnan(std_val)
                    and std_val > 0):
                score = roc_val / (std_val * math.sqrt(252))
                sig_record[f'sharpe_{name}'] = float(score)
                above[name] = score

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class DualLookbackMomentum(bt.Strategy):
    """双周期动量: 综合短期+中期ROC排名，避免单一周期噪音

    与原始动量不同: 排名 = (ROC_short + ROC_long) / 2，
    要求两个周期都有正动量才纳入。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_short', 15),
        ('roc_long', 60),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc_s': bt.indicators.RateOfChange(d.close, period=self.p.roc_short),
                'roc_l': bt.indicators.RateOfChange(d.close, period=self.p.roc_long),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_s = self.inds[name]['roc_s'][0]
            roc_l = self.inds[name]['roc_l'][0]
            px = d.close[0]
            sig_record[name] = float(roc_s) if not np.isnan(roc_s) else None
            sig_record[f'roc_l_{name}'] = float(roc_l) if not np.isnan(roc_l) else None
            if (not np.isnan(ma_val) and px > ma_val
                    and not np.isnan(roc_s) and not np.isnan(roc_l)
                    and roc_s > 0 and roc_l > 0):
                above[name] = (roc_s + roc_l) / 2

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


class TrendStrengthMomentum(bt.Strategy):
    """趋势确认动量: 要求MA本身也在上升(MA slope > 0)，避免假突破

    在原始动量基础上增加: MA当前值 > MA(ma_days)天前的值。
    这排除"价格短暂站上下跌中的均线"的假信号。T+1 执行。
    """

    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('rebalance_mode', 'daily'),
        ('min_hold', 0),
    )

    def __init__(self):
        self.inds = {}
        for d in self.datas:
            name = d._name
            self.inds[name] = {
                'ma': bt.indicators.SMA(d.close, period=self.p.ma_days),
                'roc': bt.indicators.RateOfChange(d.close, period=self.p.roc_days),
            }
        self._daily_holding = []
        self._daily_value = []
        self._daily_signals = []
        self._trade_log = []
        self._holding = None
        self._last_trade_bar = -999

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._daily_holding.append((dt, self._holding))
        self._daily_value.append((dt, self.broker.getvalue()))

        should_check = (
            self.p.rebalance_mode == 'daily' or dt.weekday() == 4
        )
        if not should_check:
            return

        above = {}
        sig_record = {'_dt': dt}
        for d in self.datas:
            name = d._name
            if d.openinterest[0] == 0:
                continue
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            px = d.close[0]
            sig_record[name] = float(roc_val) if not np.isnan(roc_val) else None

            ma_rising = False
            if len(self) > self.p.ma_days:
                ma_past = self.inds[name]['ma'][-self.p.ma_days]
                ma_rising = not np.isnan(ma_past) and ma_val > ma_past
            sig_record[f'ma_rising_{name}'] = ma_rising

            if (not np.isnan(ma_val) and px > ma_val
                    and not np.isnan(roc_val) and ma_rising):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None
        sig_record['holding'] = new_holding if above else None
        self._daily_signals.append(sig_record)

        if new_holding != self._holding \
                and len(self) - self._last_trade_bar >= self.p.min_hold:
            _execute_trade(self, dt, new_holding)


STRATEGIES = {
    'momentum': MomentumRotation,
    'rsi': RSIMeanReversion,
    'bb': BollingerBandsRotation,
    'macd': MACDTrendFollowing,
    'mom_rsi': MomentumRSI,
    'mom_bb': MomentumBB,
    'vol_weighted': VolWeightedMomentum,
    'dual_lookback': DualLookbackMomentum,
    'trend_strength': TrendStrengthMomentum,
    'stop_loss': StopLossMomentum,
    'moc': MOCRotation,
    'moo': MOORotation,
}


def _setup_cerebro(prices, mode, ma_days, roc_days, min_hold=0, strategy='momentum',
                   open_prices=None, exec_mode='moc', start_date=None, end_date=None):
    """创建并配置 Cerebro 实例

    exec_mode: 'moc' (broker.set_coc=True) 或 'moo' (cheat_on_open + broker.set_coo)
    """
    if exec_mode == 'moo':
        cerebro = bt.Cerebro(cheat_on_open=True)
        cerebro.broker.set_coo(True)
    else:
        cerebro = bt.Cerebro()
        cerebro.broker.set_coc(True)  # MOC: orders fill at same bar's close

    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.addcommissioninfo(StampDutyCommission())

    # Pre-compute MA/ROC with same function as manual engine
    ma_df, roc_df, _ = calc_indicators(prices, ma_days, roc_days)

    etf_names = list(prices.columns)
    for name in etf_names:
        data = _make_data_feed(prices, name, open_prices=open_prices,
                               ma_df=ma_df, roc_df=roc_df)
        cerebro.adddata(data)

    strat_cls = STRATEGIES[strategy]
    strat_kwargs = dict(etf_names=etf_names, rebalance_mode=mode, min_hold=min_hold)
    if strategy in ('moc', 'moo'):
        strat_kwargs['start_date'] = start_date
        strat_kwargs['end_date'] = end_date
        strat_kwargs['ma_df'] = ma_df
        strat_kwargs['roc_df'] = roc_df
    if strategy in ('momentum', 'mom_rsi', 'mom_bb', 'vol_weighted', 'trend_strength', 'stop_loss',
                    'moc', 'moo'):
        strat_kwargs['ma_days'] = ma_days
        strat_kwargs['roc_days'] = roc_days
    elif strategy == 'dual_lookback':
        strat_kwargs['ma_days'] = ma_days
    cerebro.addstrategy(strat_cls, **strat_kwargs)

    return cerebro


def _convert_output(strat, prices, start_date, end_date, etf_names):
    """将 backtrader 策略输出转换为与原接口一致的数据结构"""
    value_map = {dt: v for dt, v in strat._daily_value}
    nav_raw = pd.Series(value_map, name='nav').sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    nav = nav_raw[(nav_raw.index >= start_ts) & (nav_raw.index <= end_ts)]
    if len(nav) == 0:
        nav = pd.Series(1.0, index=[end_ts])
    nav = nav / nav.iloc[0]

    ret = nav.pct_change().fillna(0.0)

    price_trim = (prices.index >= start_ts) & (prices.index <= end_ts)
    returns = prices.pct_change(fill_method=None)
    bench_ret = returns[price_trim].mean(axis=1)
    bench_nav = (1 + bench_ret).cumprod()

    trade_details = [t for t in strat._trade_log
                     if start_ts <= pd.Timestamp(t[0]) <= end_ts]
    trades = len(trade_details)
    trade_dates = [pd.Timestamp(t[0]) for t in trade_details]

    return nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details, strat._daily_signals


def run_backtest_bt(prices, mode, start_date, end_date, ma_days=60, roc_days=25, min_hold=0,
                    strategy='moc', open_prices=None, exec_mode='moc'):
    """回测接口。Backtrader 引擎 — 信号和收益与原手动引擎 100% 一致。

    核心设计:
    1. 数据从回测起始日 (start_date) 开始, 天然无 warmup 仓位
    2. MA/ROC 预计算注入数据源, 与手动引擎完全一致
    3. coc=True: 订单在同 bar 收盘价成交 (MOC); coo=True: 在 next_open 开盘价成交 (MOO)
    4. NAV 直接从 broker.getvalue() 提取, 无需清理 warmup 仓位

    返回 (nav, bench_nav, ret, bench_ret, trades, trade_details, daily_signals)
    """
    from etf_data import calc_indicators as _calc

    etf_names = list(prices.columns)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Slice data from start_date (Backtrader has no old positions)
    prices_bt = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if len(prices_bt) < ma_days:
        # short period: use full data but will trim output
        prices_bt = prices[prices.index >= start_ts - pd.Timedelta(days=5*365)]

    # Pre-compute MA/ROC on FULL data, then align to BT range
    ma_full, roc_full, _ = _calc(prices, ma_days, roc_days)
    ma_bt = ma_full.loc[prices_bt.index]
    roc_bt = roc_full.loc[prices_bt.index]

    # MOO: Backtrader coo has known timing issues with multi-asset rotation.
    # Delegate to proven manual engine for correct overnight+intraday execution.
    if exec_mode == 'moo':
        _fn = _get_manual_bt()
        if _fn:
            return _fn(prices, mode, start_date, end_date, ma_days, roc_days, min_hold,
                       open_prices=open_prices, delay=0)
        raise RuntimeError("Manual run_backtest not available for MOO fallback")

    # MOC: Backtrader coc with date-indexed pre-computed MA/ROC. Verified.
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.addcommissioninfo(StampDutyCommission())

    for name in etf_names:
        data = _make_data_feed(prices_bt, name, ma_df=ma_bt, roc_df=roc_bt,
                               open_prices=open_prices)
        cerebro.adddata(data)

    strat_cls = MOCRotation if exec_mode == 'moc' else MOORotation
    strat_kwargs = dict(etf_names=etf_names, rebalance_mode=mode, min_hold=min_hold,
                        ma_df=ma_full, roc_df=roc_full, start_date=None, end_date=None)
    cerebro.addstrategy(strat_cls, **strat_kwargs)
    results = cerebro.run()
    strat = results[0]
    return _convert_output(strat, prices_bt, start_date, end_date, etf_names)


def position_dist_bt(prices, start_date, end_date, mode, ma_days=60, roc_days=25, min_hold=0,
                     strategy='moc', open_prices=None, exec_mode='moc'):
    """与原 position_dist() 签名和返回值完全一致

    返回 (持有天数dict, 买入次数dict, 收益占比dict, 持有期累计收益dict, 胜率dict)
    """
    etf_names = list(prices.columns)
    cerebro = _setup_cerebro(prices, mode, ma_days, roc_days, min_hold, strategy,
                             start_date=start_date, end_date=end_date,
                             open_prices=open_prices, exec_mode=exec_mode)
    results = cerebro.run()
    strat = results[0]

    daily_ret = prices.pct_change(fill_method=None)
    COMMISSION_RATE = 0.0001 + 0.0005

    days = {n: 0 for n in etf_names}
    days["CASH"] = 0
    buys = {n: 0 for n in etf_names}
    nav = {n: 1.0 for n in etf_names}
    pos_days = {n: 0 for n in etf_names}
    hold_days_for_wr = {n: 0 for n in etf_names}
    log_ret = {n: 0.0 for n in etf_names}
    log_ret["CASH"] = 0.0

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for dt, from_etf, to_etf in strat._trade_log:
        if to_etf is not None and start_ts <= pd.Timestamp(dt) <= end_ts:
            buys[to_etf] += 1

    holding_map = {dt: h for dt, h in strat._daily_holding}
    for i in range(len(prices)):
        dt = prices.index[i]
        in_range = dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date)
        if not in_range:
            continue

        h = holding_map.get(dt)
        if h is None:
            h = "CASH"
        days[h] += 1
        if h != "CASH":
            r = daily_ret[h].iloc[i]
            if not pd.isna(r):
                nav[h] *= (1 + r)
                hold_days_for_wr[h] += 1
                if r > 0:
                    pos_days[h] += 1
                log_ret[h] += math.log(1 + r)

    for dt, from_etf, to_etf in strat._trade_log:
        if dt >= pd.Timestamp(start_date) and dt <= pd.Timestamp(end_date):
            if from_etf is not None and from_etf in log_ret:
                log_ret[from_etf] += math.log(1 - COMMISSION_RATE)

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
