"""ETF双动量轮动 — backtrader 引擎

提供与原手写回测完全相同的接口，通过 backtrader 执行回测逻辑。
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

import backtrader as bt

from etf_data import calc_indicators


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


def _make_data_feed(prices, name):
    """创建适配仅含收盘价的ETF数据，open/high/low 复用 close"""
    df = pd.DataFrame(index=prices.index)
    df['open'] = prices[name].values
    df['high'] = prices[name].values
    df['low'] = prices[name].values
    df['close'] = prices[name].values
    df['volume'] = 100
    data = bt.feeds.PandasData(dataname=df)
    data._name = name
    return data


class MomentumRotation(bt.Strategy):
    """双动量轮动: MA60趋势过滤 + ROC20动量排名，持有最优单一ETF"""
    params = (
        ('etf_names', []),
        ('ma_days', 60),
        ('roc_days', 25),
        ('rebalance_mode', 'daily'),  # 'daily' | 'friday'
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

        self._daily_holding = []   # (dt, etf_name|None)
        self._daily_value = []     # (dt, portfolio_value)
        self._trade_log = []       # (dt, from_etf|None, to_etf|None)
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
        if self.p.min_hold > 0 and len(self) - self._last_trade_bar < self.p.min_hold:
            return

        # 动量过滤 + 排名
        above = {}
        for i, d in enumerate(self.datas):
            name = d._name
            ma_val = self.inds[name]['ma'][0]
            roc_val = self.inds[name]['roc'][0]
            px = d.close[0]
            if not np.isnan(ma_val) and px > ma_val and not np.isnan(roc_val):
                above[name] = roc_val

        new_holding = max(above, key=above.get) if above else None

        if new_holding != self._holding:
            # 平仓当前持仓
            if self._holding is not None:
                for d in self.datas:
                    if d._name == self._holding:
                        self.close(data=d)
                        break
            # 开仓新标的
            if new_holding is not None:
                for d in self.datas:
                    if d._name == new_holding:
                        self.order_target_percent(d, target=0.995)
                        break

            self._trade_log.append((dt, self._holding, new_holding))
            self._holding = new_holding
            self._last_trade_bar = len(self)


def _setup_cerebro(prices, mode, ma_days, roc_days, min_hold=0):
    """创建并配置 Cerebro 实例"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000.0)  # 大额现金避免整数股截断

    # 添加自定义佣金
    cerebro.broker.addcommissioninfo(StampDutyCommission())

    # 添加各ETF数据
    etf_names = list(prices.columns)
    for name in etf_names:
        data = _make_data_feed(prices, name)
        cerebro.adddata(data)

    cerebro.addstrategy(
        MomentumRotation,
        etf_names=etf_names,
        ma_days=ma_days,
        roc_days=roc_days,
        rebalance_mode=mode,
        min_hold=min_hold,
    )

    return cerebro


def _convert_output(strat, prices, start_date, end_date, etf_names):
    """将 backtrader 策略输出转换为与原接口一致的数据结构"""
    # 从策略日志构建 NAV Series
    value_map = {dt: v for dt, v in strat._daily_value}
    nav_raw = pd.Series(value_map, name='nav').sort_index()
    # 截取回测区间
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    nav = nav_raw[(nav_raw.index >= start_ts) & (nav_raw.index <= end_ts)]
    if len(nav) == 0:
        nav = pd.Series(1.0, index=[end_ts])
    nav = nav / nav.iloc[0]

    ret = nav.pct_change().fillna(0.0)

    # 基准: 等权平均（与原逻辑一致），使用 prices 自带的日期索引截取
    price_trim = (prices.index >= start_ts) & (prices.index <= end_ts)
    returns = prices.pct_change(fill_method=None)
    bench_ret = returns[price_trim].mean(axis=1)
    bench_nav = (1 + bench_ret).cumprod()

    # 交易记录
    trade_dates = [t[0] for t in strat._trade_log]
    trade_details = strat._trade_log.copy()
    trades = len(strat._trade_log)

    return nav, bench_nav, ret, bench_ret, trades, trade_dates, trade_details


def run_backtest_bt(prices, mode, start_date, end_date, ma_days=60, roc_days=25, min_hold=0):
    """与原 run_backtest() 签名和返回值完全一致"""
    etf_names = list(prices.columns)
    cerebro = _setup_cerebro(prices, mode, ma_days, roc_days, min_hold)
    results = cerebro.run()
    strat = results[0]
    return _convert_output(strat, prices, start_date, end_date, etf_names)


def position_dist_bt(prices, start_date, end_date, mode, ma_days=60, roc_days=25, min_hold=0):
    """与原 position_dist() 签名和返回值完全一致

    返回 (持有天数dict, 买入次数dict, 收益占比dict, 持有期累计收益dict, 胜率dict)
    """
    etf_names = list(prices.columns)
    cerebro = _setup_cerebro(prices, mode, ma_days, roc_days, min_hold)
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

    # 从交易日志重建买入次数
    for dt, from_etf, to_etf in strat._trade_log:
        if to_etf is not None:
            buys[to_etf] += 1

    # 遍历回测区间，统计各ETF持仓收益
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

    # 佣金影响: 从 trade_log 推算
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
