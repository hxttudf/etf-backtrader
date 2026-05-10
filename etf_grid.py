"""网格交易引擎 — 东方财富动态基准模型

每次触发后基准价移到触发价，网格整体移动。
买入在 P → 基准价=P → 新网格以 P 为中心 → 下一格 P+step 待卖
"""

from dataclasses import dataclass, field
from typing import Literal
import math
import numpy as np
import pandas as pd

GridType = Literal["arithmetic", "geometric", "volatility"]


@dataclass(frozen=True)
class Trade:
    datetime: pd.Timestamp
    side: str
    price: float
    amount: float
    quantity: int


@dataclass
class GridConfig:
    symbol: str
    grid_type: GridType = "arithmetic"
    n_levels: int = 10
    step_value: float = 0.0    # 价差(arithmetic)或百分比%(geometric)
    base_price: float = 0.0    # 初始基准价(0=首日收盘)
    amount_per_grid: float = 10000.0
    max_positions: int = 10
    initial_capital: float = 0.0
    commission: float = 0.0003
    slippage: float = 0.001


class GridEngine:
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.cash: float = 0
        self.position: int = 0
        self.today_bought: int = 0
        self.base_price: float = 0
        self.trades: list = []
        self._pending_sell: float | None = None  # 当前待卖的触发价

    def _step(self, price: float) -> float:
        """计算一格步长（绝对金额）"""
        cfg = self.cfg
        if cfg.grid_type == "geometric" and cfg.step_value > 0:
            return price * cfg.step_value / 100
        return cfg.step_value if cfg.step_value > 0 else price * 0.01

    def _buy_price(self) -> float:
        """当前买入触发价 = 基准价 - 一格"""
        return round(self.base_price - self._step(self.base_price), 4)

    def _sell_price(self) -> float:
        """当前卖出触发价 = 基准价 + 一格"""
        return round(self.base_price + self._step(self.base_price), 4)

    def _on_trade(self):
        """触发后以触发价为新基准价，重建待卖"""
        pass  # base_price 已在 buy/sell 中更新

    def _process_bar(self, ohlc: pd.Series, dt: pd.Timestamp):
        o, h, l = float(ohlc["open"]), float(ohlc["high"]), float(ohlc["low"])
        cfg = self.cfg

        # T+1 重置
        if self.trades and isinstance(dt, pd.Timestamp):
            last_dt = self.trades[-1].datetime
            if isinstance(last_dt, pd.Timestamp) and last_dt.date() < dt.date():
                self.today_bought = 0

        bp = self._buy_price()
        sp = self._sell_price()

        # ── 开→高：向上穿越卖出价 → 卖出 ──
        if self._pending_sell is not None and self.position - self.today_bought > 0:
            if o < sp <= h:
                qty = int(cfg.amount_per_grid / sp)
                if qty > 0:
                    qty = min(qty, self.position - self.today_bought)
                    rev = qty * sp * (1 - cfg.commission) - 0.1
                    self.cash += rev
                    self.position -= qty
                    self.trades.append(Trade(dt, "sell", sp, rev, qty))
                    self.base_price = sp
                    self._pending_sell = None

        # ── 高→低：向下穿越买入价 → 买入 ──
        if self._pending_sell is None:  # 没有待卖才能再买
            if l <= bp < h:
                cost = cfg.amount_per_grid
                qty = int(cost / (bp * (1 + cfg.slippage)))
                if qty > 0 and self.cash >= qty * bp * (1 + cfg.commission) + 0.1:
                    total = qty * bp * (1 + cfg.commission) + 0.1
                    self.cash -= total
                    self.position += qty
                    self.today_bought += qty
                    self.trades.append(Trade(dt, "buy", bp, total, qty))
                    self.base_price = bp
                    self._pending_sell = self._sell_price()

    def run(self, df: pd.DataFrame) -> list[Trade]:
        if len(df) == 0:
            return []
        cfg = self.cfg
        first_close = float(df["close"].iloc[0])
        self.base_price = cfg.base_price if cfg.base_price > 0 else first_close
        cap = cfg.initial_capital if cfg.initial_capital > 0 else (
            cfg.amount_per_grid * cfg.max_positions)
        self.cash = cap
        self.position = 0
        self.today_bought = 0
        self._pending_sell = None
        self.trades = []

        for dt, row in df.iterrows():
            self._process_bar(row, dt)

        return self.trades

    def get_metrics(self, df: pd.DataFrame) -> dict:
        if not self.trades:
            return {"总收益": 0, "买入次数": 0, "卖出次数": 0}
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else (
            self.cfg.amount_per_grid * self.cfg.max_positions)
        fv = self.cash + self.position * float(df["close"].iloc[-1])
        buys = [t for t in self.trades if t.side == "buy"]
        sells = [t for t in self.trades if t.side == "sell"]
        wins = sum(1 for s in sells for b in buys[:1] if s.price > b.price)
        pairs = min(len(buys), len(sells) if sells else 0)
        nav_s = df["close"] * self.position + self.cash
        dd = (nav_s / nav_s.cummax() - 1).min() if len(nav_s) > 0 else 0
        return {
            "总收益": fv / cap - 1, "买入次数": len(buys), "卖出次数": len(sells),
            "胜率": wins / pairs if pairs > 0 else 0, "最大回撤": dd,
            "初始资金": cap, "最终资产": fv,
            "持仓份额": self.position, "剩余现金": self.cash,
        }

    def get_nav_series(self, df: pd.DataFrame) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else (
            self.cfg.amount_per_grid * self.cfg.max_positions)
        dates = sorted(set(d.date() for d in df.index))
        pos, cash = 0, cap
        tm = {}
        for t in self.trades:
            d = t.datetime.date()
            if d not in tm:
                tm[d] = [0, 0.0]
            if t.side == "buy":
                tm[d][0] += t.quantity
                tm[d][1] -= t.amount
            else:
                tm[d][0] -= t.quantity
                tm[d][1] += t.amount
        nv = {}
        for d in dates:
            if d in tm:
                pos += tm[d][0]
                cash += tm[d][1]
            cd = df[df.index.date == d]
            close = float(cd["close"].iloc[-1]) if len(cd) > 0 else 0
            nv[d] = (cash + pos * close) / cap
        return pd.Series(nv, name="nav").sort_index()


def run_grid_backtest(symbol, df, grid_type="arithmetic",
                      n_levels=10, step_value=0.0,
                      amount_per_grid=10000.0, max_positions=10,
                      initial_capital=0.0, initial_shares=0,
                      base_price=0.0, commission=0.0003, slippage=0.001):
    config = GridConfig(
        symbol=symbol, grid_type=grid_type, n_levels=n_levels,
        step_value=step_value, base_price=base_price,
        amount_per_grid=amount_per_grid, max_positions=max_positions,
        initial_capital=initial_capital, commission=commission, slippage=slippage,
    )
    engine = GridEngine(config)
    trades = engine.run(df)
    metrics = engine.get_metrics(df)
    return trades, metrics, engine
