"""网格交易引擎 — 纯 pandas 实现

核心逻辑（东方财富网格模型）：
  价格向下穿越 L_i → 买入, 在 L_{i+1} 设待卖
  价格向上穿越 L_i → 如果有待卖, 卖出（平上一格的仓，赚相邻两格差价）

参考：
  - yansongwel/etf-quant: Signal 模型
  - jorben/grider: 网格计算器
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
    side: str           # "buy" | "sell"
    price: float
    amount: float
    quantity: int
    grid_idx: int       # buy: 买入格; sell: 平仓的买入格


@dataclass
class GridState:
    levels: list           # [(price, buy_count, sell_count), ...]
    pending_sells: dict    # level_idx -> 待卖数（在 idx 格卖出 idx-1 的买入）
    cash: float
    position: int
    today_bought: int
    base_price: float
    total_value: float
    trades: list = field(default_factory=list)


@dataclass
class GridConfig:
    symbol: str
    grid_type: GridType = "arithmetic"
    n_levels: int = 10
    step_value: float = 0.0
    price_low: float = 0.0
    price_high: float = 0.0
    base_price: float = 0.0
    amount_per_grid: float = 10000.0
    max_positions: int = 10
    initial_capital: float = 0.0
    initial_shares: int = 0
    commission: float = 0.0003
    slippage: float = 0.001
    atr_period: int = 20
    atr_multiplier: float = 2.0


def arithmetic_levels(price_low, price_high, n_levels, base_price):
    step = (price_high - price_low) / max(n_levels - 1, 1)
    return [price_low + i * step for i in range(n_levels)]


def geometric_levels(price_low, price_high, n_levels, base_price):
    if price_low <= 0 or price_high <= 0:
        return arithmetic_levels(price_low, price_high, n_levels, base_price)
    ratio = (price_high / price_low) ** (1 / max(n_levels - 1, 1))
    return [price_low * (ratio ** i) for i in range(n_levels)]


def volatility_levels(price_low, price_high, n_levels, base_price,
                      df=None, atr_period=20, atr_multiplier=2.0):
    if df is None or len(df) < atr_period + 1:
        return arithmetic_levels(price_low, price_high, n_levels, base_price)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-atr_period:]) if len(tr) >= atr_period else np.mean(tr)
    step = atr * atr_multiplier
    prices = [base_price + (i - n_levels // 2) * step for i in range(n_levels)]
    prices = [p for p in prices if price_low <= p <= price_high]
    while len(prices) < n_levels:
        if prices[0] > price_low:
            prices.insert(0, prices[0] - step)
        elif prices[-1] < price_high:
            prices.append(prices[-1] + step)
        else:
            break
    return prices[:n_levels]


GRID_CALCULATORS = {
    "arithmetic": arithmetic_levels,
    "geometric": geometric_levels,
    "volatility": volatility_levels,
}


class GridEngine:
    """网格交易回测引擎（东方财富模型）"""

    def __init__(self, config: GridConfig):
        self.cfg = config
        self.state: GridState | None = None

    def _compute_n_levels(self) -> int:
        cfg = self.cfg
        if cfg.step_value > 0 and cfg.price_high > cfg.price_low:
            if cfg.grid_type == "geometric":
                ratio = 1 + cfg.step_value / 100
                n = int(math.log(cfg.price_high / cfg.price_low) / math.log(ratio)) + 1
            else:
                n = int((cfg.price_high - cfg.price_low) / cfg.step_value) + 1
            return max(n, 2)
        return cfg.n_levels

    def _init_state(self, first_price: float, df: pd.DataFrame | None = None):
        cfg = self.cfg
        n_levels = self._compute_n_levels()
        if cfg.base_price > 0:
            bp = cfg.base_price
            if cfg.grid_type == "geometric" and cfg.step_value > 0:
                ratio = 1 + cfg.step_value / 100
                half_n = n_levels // 2
                pl = bp * (ratio ** -half_n)
                ph = bp * (ratio ** half_n)
            elif cfg.step_value > 0:
                half_n = n_levels // 2
                pl = bp - half_n * cfg.step_value
                ph = bp + half_n * cfg.step_value
            else:
                pl = cfg.price_low if cfg.price_low > 0 else bp * 0.9
                ph = cfg.price_high if cfg.price_high > 0 else bp * 1.1
        else:
            pl = cfg.price_low if cfg.price_low > 0 else (float(df["low"].min()) if df is not None else first_price * 0.9)
            ph = cfg.price_high if cfg.price_high > 0 else (float(df["high"].max()) if df is not None else first_price * 1.1)
        calc = GRID_CALCULATORS[cfg.grid_type]
        if cfg.grid_type == "volatility":
            prices = calc(pl, ph, n_levels, first_price, df=df,
                          atr_period=cfg.atr_period, atr_multiplier=cfg.atr_multiplier)
        else:
            prices = calc(pl, ph, n_levels, first_price)
        levels = [[round(p, 4), 0, 0] for p in sorted(prices)]
        total_capital = cfg.initial_capital if cfg.initial_capital > 0 else (
            cfg.amount_per_grid * min(cfg.max_positions, n_levels))
        init_cost = cfg.initial_shares * first_price
        self.state = GridState(
            levels=levels, cash=total_capital - init_cost,
            position=cfg.initial_shares, today_bought=0,
            base_price=first_price, total_value=total_capital,
            pending_sells={}, trades=[],
        )

    def _buy_at(self, idx: int, price: float, dt: pd.Timestamp):
        """在 L_idx 买入，设待卖于 L_{idx+1}"""
        s = self.state
        cfg = self.cfg
        if sum(s.pending_sells.values()) >= cfg.max_positions:
            return
        cost = cfg.amount_per_grid
        qty = int(cost / (price * (1 + cfg.slippage)))
        if qty <= 0:
            return
        total = qty * price * (1 + cfg.commission) + 0.1
        if s.cash < total:
            return
        s.cash -= total
        s.position += qty
        s.today_bought += qty
        s.levels[idx][1] += 1
        s.total_value = s.cash + s.position * price
        s.base_price = price
        s.trades.append(Trade(dt, "buy", price, total, qty, idx))
        if idx + 1 < len(s.levels):
            s.pending_sells[idx + 1] = s.pending_sells.get(idx + 1, 0) + 1

    def _sell_at(self, idx: int, price: float, dt: pd.Timestamp):
        """在 L_idx 卖出（平 L_{idx-1} 的买入）"""
        s = self.state
        cfg = self.cfg
        if s.pending_sells.get(idx, 0) <= 0:
            return
        if s.position - s.today_bought <= 0:
            return
        qty = int(cfg.amount_per_grid / price)
        if qty <= 0:
            return
        qty = min(qty, s.position - s.today_bought)
        revenue = qty * price * (1 - cfg.commission) - 0.1
        s.cash += revenue
        s.position -= qty
        s.pending_sells[idx] = s.pending_sells[idx] - 1
        s.levels[idx - 1][2] += 1
        s.total_value = s.cash + s.position * price
        s.base_price = price
        s.trades.append(Trade(dt, "sell", price, revenue, qty, idx - 1))

    def _process_bar(self, ohlc: pd.Series, dt: pd.Timestamp):
        s = self.state
        if s is None:
            return
        o, h, l = ohlc["open"], ohlc["high"], ohlc["low"]
        prices = [x[0] for x in s.levels]
        if s.trades and isinstance(dt, pd.Timestamp):
            last_dt = s.trades[-1].datetime
            if isinstance(last_dt, pd.Timestamp) and last_dt.date() < dt.date():
                s.today_bought = 0
        # 开→高：向上穿线 → 卖出待卖
        for i in range(len(prices)):
            if o < prices[i] <= h and s.pending_sells.get(i, 0) > 0:
                self._sell_at(i, prices[i], dt)
        # 高→低：向下穿线 → 买入
        for i in range(len(prices)):
            if l <= prices[i] < h:
                self._buy_at(i, prices[i], dt)

    def run(self, df: pd.DataFrame) -> list[Trade]:
        if len(df) == 0:
            return []
        self._init_state(float(df["close"].iloc[0]), df)
        for dt, row in df.iterrows():
            self._process_bar(row, dt)
        return self.state.trades

    def get_metrics(self, df: pd.DataFrame) -> dict:
        if self.state is None or len(self.state.trades) == 0:
            return {"总收益": 0, "买入次数": 0, "卖出次数": 0}
        trades = self.state.trades
        n_lvls = self._compute_n_levels()
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else (
            self.cfg.amount_per_grid * min(self.cfg.max_positions, n_lvls))
        fv = self.state.cash + self.state.position * df["close"].iloc[-1]
        buys = {}
        wins = pairs = 0
        for t in trades:
            if t.side == "buy":
                buys[t.grid_idx] = t
            elif t.side == "sell" and t.grid_idx in buys:
                pairs += 1
                if t.price > buys[t.grid_idx].price:
                    wins += 1
        nav_s = df["close"] * self.state.position + self.state.cash
        dd = (nav_s / nav_s.cummax() - 1).min() if len(nav_s) > 0 else 0
        return {
            "总收益": fv / cap - 1, "买入次数": sum(1 for t in trades if t.side == "buy"),
            "卖出次数": sum(1 for t in trades if t.side == "sell"),
            "胜率": wins / pairs if pairs > 0 else 0, "最大回撤": dd,
            "初始资金": cap, "最终资产": fv,
            "持仓份额": self.state.position, "剩余现金": self.state.cash,
        }

    def get_nav_series(self, df: pd.DataFrame) -> pd.Series:
        if self.state is None:
            return pd.Series(dtype=float)
        cfg = self.cfg
        n_lvls = self._compute_n_levels()
        cap = cfg.initial_capital if cfg.initial_capital > 0 else (
            cfg.amount_per_grid * min(cfg.max_positions, n_lvls))
        dates = sorted(set(d.date() for d in df.index))
        if not dates:
            return pd.Series(dtype=float)
        pos = cfg.initial_shares
        cash = cap - pos * float(df.iloc[0]["close"])
        tm = {}
        for t in self.state.trades:
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

    def get_grid_df(self) -> pd.DataFrame:
        if self.state is None:
            return pd.DataFrame()
        rows = []
        for i, (price, bc, sc) in enumerate(self.state.levels):
            pending = self.state.pending_sells.get(i, 0)
            parts = []
            if bc > 0:
                parts.append(f"买{bc}")
            if pending > 0:
                parts.append(f"待{pending}")
            if sc > 0:
                parts.append(f"卖{sc}")
            rows.append({"线": f"L{i+1}", "价格": price,
                         "状态": "/".join(parts) if parts else "—",
                         "活跃": "✅" if (bc or sc or pending) else ""})
        return pd.DataFrame(rows)


def run_grid_backtest(symbol, df, grid_type="arithmetic",
                      n_levels=10, step_value=0.0,
                      amount_per_grid=10000.0, max_positions=10,
                      initial_capital=0.0, initial_shares=0,
                      base_price=0.0, commission=0.0003, slippage=0.001):
    config = GridConfig(
        symbol=symbol, grid_type=grid_type, n_levels=n_levels,
        step_value=step_value, price_low=float(df["low"].min()),
        price_high=float(df["high"].max()),
        amount_per_grid=amount_per_grid, max_positions=max_positions,
        initial_shares=initial_shares, initial_capital=initial_capital,
        base_price=base_price, commission=commission, slippage=slippage,
    )
    engine = GridEngine(config)
    trades = engine.run(df)
    metrics = engine.get_metrics(df)
    return trades, metrics, engine
