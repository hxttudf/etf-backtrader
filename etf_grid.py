"""网格交易引擎 — 纯 pandas 实现，无 backtrader 依赖

支持 3 种网格类型（等差/等比/ATR 动态），逐 K 线高精度穿越检测，
严格 A 股 T+1 规则。

参考：
  - yansongwel/etf-quant: Signal 模型、T+1 执行
  - jorben/grider: ArithmeticGridCalculator / GeometricGridCalculator
"""

from dataclasses import dataclass, field
from typing import Literal
import math

import numpy as np
import pandas as pd

# ── 网格类型 ──────────────────────────────────────────────────

GridType = Literal["arithmetic", "geometric", "volatility"]


# ── 数据模型 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Trade:
    datetime: pd.Timestamp
    side: str          # "buy" | "sell"
    price: float
    amount: float      # 成交金额
    quantity: int      # 份额
    grid_idx: int      # 触发的网格索引


@dataclass
class GridState:
    """网格运行时状态"""
    levels: list       # [(price, bought, sold), ...]
    cash: float
    position: int      # 总持仓
    today_bought: int  # 今日买入（T+1 约束）
    base_price: float
    total_value: float
    trades: list = field(default_factory=list)


@dataclass
class GridConfig:
    """网格配置"""
    symbol: str
    grid_type: GridType = "arithmetic"
    n_levels: int = 10        # 网格线数（step_value=0 时使用）
    step_value: float = 0.0   # 每格步长：价差(arithmetic)或百分比(geometric)。设>0时覆盖n_levels
    price_low: float = 0.0
    price_high: float = 0.0
    amount_per_grid: float = 10000.0
    max_positions: int = 10
    initial_capital: float = 0.0  # 总本金（0=自动按 amount_per_grid*max_positions）
    initial_shares: int = 0     # 初始底仓（股/份）
    commission: float = 0.0003
    slippage: float = 0.001
    atr_period: int = 20
    atr_multiplier: float = 2.0


# ── 网格计算器 ──────────────────────────────────────────────

def arithmetic_levels(price_low: float, price_high: float,
                      n_levels: int, base_price: float) -> list[float]:
    """等差网格：每格固定差价"""
    step = (price_high - price_low) / (n_levels - 1) if n_levels > 1 else 0.01
    prices = [price_low + i * step for i in range(n_levels)]
    return prices


def geometric_levels(price_low: float, price_high: float,
                     n_levels: int, base_price: float) -> list[float]:
    """等比网格：每格固定百分比"""
    if price_low <= 0 or price_high <= 0:
        return arithmetic_levels(price_low, price_high, n_levels, base_price)
    ratio = (price_high / price_low) ** (1 / max(n_levels - 1, 1))
    prices = [price_low * (ratio ** i) for i in range(n_levels)]
    return prices


def volatility_levels(price_low: float, price_high: float,
                      n_levels: int, base_price: float,
                      df: pd.DataFrame | None = None,
                      atr_period: int = 20,
                      atr_multiplier: float = 2.0) -> list[float]:
    """ATR 动态网格：格距 = ATR × 倍数"""
    if df is None or len(df) < atr_period + 1:
        return arithmetic_levels(price_low, price_high, n_levels, base_price)

    # 计算 ATR
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(high[1:] - low[1:],
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-atr_period:]) if len(tr) >= atr_period else np.mean(tr)
    step = atr * atr_multiplier

    # 以 base_price 为中心，上下展开
    half_range = step * (n_levels // 2)
    prices = [base_price + (i - n_levels // 2) * step for i in range(n_levels)]
    prices = [p for p in prices if price_low <= p <= price_high]
    # 如果展开不够，补齐
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


# ── 网格引擎 ──────────────────────────────────────────────

class GridEngine:
    """网格交易回测引擎"""

    def __init__(self, config: GridConfig):
        self.cfg = config
        self.state: GridState | None = None

    def _compute_n_levels(self) -> int:
        """根据 step_value 或 n_levels 计算网格线数"""
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
        """初始化网格和状态"""
        cfg = self.cfg
        n_levels = self._compute_n_levels()
        calc = GRID_CALCULATORS[cfg.grid_type]

        if cfg.grid_type == "volatility":
            prices = calc(cfg.price_low, cfg.price_high, n_levels, first_price,
                          df=df, atr_period=cfg.atr_period, atr_multiplier=cfg.atr_multiplier)
        else:
            prices = calc(cfg.price_low, cfg.price_high, n_levels, first_price)

        levels = [(round(p, 4), False, False) for p in sorted(prices)]
        # 本金：设了 initial_capital 就用它，否则按 amount_per_grid * max_positions
        total_capital = cfg.initial_capital if cfg.initial_capital > 0 else (
            cfg.amount_per_grid * min(cfg.max_positions, len(levels)))
        init_position = cfg.initial_shares
        init_cost = init_position * first_price
        self.state = GridState(
            levels=[list(l) for l in levels],
            cash=total_capital - init_cost,
            position=init_position,
            today_bought=0,
            base_price=first_price,
            total_value=total_capital,
            trades=[],
        )

    def _find_grid_idx(self, price: float) -> int:
        """找到价格所在的网格区间索引"""
        prices = [l[0] for l in self.state.levels]
        for i in range(len(prices)):
            if price < prices[i]:
                return i
        return len(prices) - 1

    def _try_buy(self, price: float, idx: int, dt: pd.Timestamp):
        """在指定网格线买入"""
        s = self.state
        cfg = self.cfg
        _, bought, _ = s.levels[idx]
        if bought:
            return
        cost = cfg.amount_per_grid
        quantity = int(cost / (price * (1 + cfg.slippage)))
        if quantity <= 0:
            return
        actual_cost = quantity * price
        commission = max(actual_cost * cfg.commission, 0.1)
        total_cost = actual_cost + commission

        if s.cash < total_cost:
            return

        s.cash -= total_cost
        s.position += quantity
        s.today_bought += quantity
        s.levels[idx][1] = True  # marked bought
        s.total_value = s.cash + s.position * price
        s.base_price = price
        s.trades.append(Trade(dt, "buy", price, total_cost, quantity, idx))

    def _try_sell(self, price: float, idx: int, dt: pd.Timestamp):
        """在指定网格线卖出"""
        s = self.state
        cfg = self.cfg
        _, bought, sold = s.levels[idx]
        if not bought or sold:
            return

        # T+1: 只能卖出昨日持仓
        available = s.position - s.today_bought
        if available <= 0:
            return

        quantity = int(cfg.amount_per_grid / price)
        if quantity <= 0:
            return
        quantity = min(quantity, available)

        revenue = quantity * price
        commission = max(revenue * cfg.commission, 0.1)
        net_revenue = revenue - commission

        s.cash += net_revenue
        s.position -= quantity
        s.levels[idx][2] = True  # marked sold
        s.total_value = s.cash + s.position * price
        s.base_price = price
        s.trades.append(Trade(dt, "sell", price, net_revenue, quantity, idx))

    def _process_bar(self, ohlc: pd.Series, dt: pd.Timestamp):
        """处理一根 K 线：按 开→高→低→收 路径检测穿越"""
        s = self.state
        if s is None:
            return

        o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
        prices = [l[0] for l in s.levels]

        # 新的一天 → 重置 today_bought
        if isinstance(dt, pd.Timestamp):
            date_key = dt.date()
            last_trade = s.trades[-1].datetime if s.trades else None
            if last_trade is not None and last_trade.date() < date_key:
                s.today_bought = 0

        # ── open → high：向上检测 ──
        current = o
        direction = 1 if h > o else -1
        if direction > 0:
            for i in range(len(prices)):
                if prices[i] > current and prices[i] <= h and not s.levels[i][1]:
                    self._try_buy(prices[i], i, dt)
        else:
            for i in reversed(range(len(prices))):
                if prices[i] < current and prices[i] >= h and not s.levels[i][2]:
                    self._try_sell(prices[i], i, dt)

        # ── high → low：先上后下 ──
        for i in reversed(range(len(prices))):
            if prices[i] < h and prices[i] >= l and not s.levels[i][2]:
                if s.levels[i][1]:
                    self._try_sell(prices[i], i, dt)
        for i in range(len(prices)):
            if prices[i] > l and prices[i] <= h and not s.levels[i][1]:
                if not s.levels[i][1]:
                    self._try_buy(prices[i], i, dt)

    def run(self, df: pd.DataFrame) -> list[Trade]:
        """运行回测

        Args:
            df: DataFrame index=datetime, columns=['open','high','low','close','volume']

        Returns:
            trade 列表
        """
        if len(df) == 0:
            return []

        first_close = df["close"].iloc[0]
        self._init_state(first_close, df)

        for dt, row in df.iterrows():
            self._process_bar(row, dt)

        return self.state.trades

    def get_metrics(self, df: pd.DataFrame) -> dict:
        """计算收益指标"""
        if self.state is None or len(self.state.trades) == 0:
            return {"总收益": 0, "交易次数": 0}

        trades = self.state.trades
        n_lvls = self._compute_n_levels()
        total_capital = self.cfg.initial_capital if self.cfg.initial_capital > 0 else (
            self.cfg.amount_per_grid * min(self.cfg.max_positions, n_lvls))
        final_value = self.state.cash + self.state.position * df["close"].iloc[-1]
        total_return = final_value / total_capital - 1

        # 按交易日计算净值
        dates = sorted(set(t.datetime.date() for t in trades))
        if len(df) > 0 and hasattr(df.index, 'date'):
            all_dates = sorted(set(d.date() for d in df.index))
        else:
            all_dates = dates

        # 胜率（按网格对计算）
        buy_map = {}
        win_count = 0
        total_pairs = 0
        for t in trades:
            if t.side == "buy":
                buy_map[t.grid_idx] = t
            elif t.side == "sell" and t.grid_idx in buy_map:
                total_pairs += 1
                buy_t = buy_map[t.grid_idx]
                if t.price > buy_t.price:
                    win_count += 1

        win_rate = win_count / total_pairs if total_pairs > 0 else 0

        # 最大回撤（按日）
        day_values = {}
        for t in trades:
            d = t.datetime.date()
            day_values[d] = None  # mark days with trades

        if len(df) > 0:
            nav = df["close"] * self.state.position + self.state.cash
        else:
            nav = pd.Series([final_value])

        peak = nav.cummax()
        dd = (nav / peak - 1).min() if len(nav) > 0 else 0

        return {
            "总收益": total_return,
            "交易次数": len([t for t in trades if t.side == "buy"]),
            "胜率": win_rate,
            "最大回撤": dd,
            "最终资产": final_value,
            "持仓份额": self.state.position,
            "剩余现金": self.state.cash,
        }

    def get_nav_series(self, df: pd.DataFrame) -> pd.Series:
        """计算每日净值序列"""
        if self.state is None:
            return pd.Series(dtype=float)
        total_capital = self.cfg.initial_capital if self.cfg.initial_capital > 0 else (
            self.cfg.amount_per_grid * min(self.cfg.max_positions, self._compute_n_levels()))
        # 建一个日期→持仓映射
        daily_pos = {}
        daily_cash = {}
        pos, cash = self.state.position, self.state.cash
        # 从 trade 记录回溯
        trade_map = {}  # date -> (new_pos, new_cash)
        for t in reversed(self.state.trades):
            trade_map[t.datetime.date()] = (t.quantity, t.amount, t.side)
        # 从末尾往前遍历
        dates = sorted(set(d.date() for d in df.index))
        for d in reversed(dates):
            daily_pos[d] = pos
            daily_cash[d] = cash
            if d in trade_map:
                qty, amt, side = trade_map[d]
                if side == "buy":
                    pos -= qty
                    cash += amt
                else:
                    pos += qty
                    cash -= amt
        nav = pd.Series({d: daily_cash[d] + daily_pos[d] * float(
            df[df.index.date == d]["close"].iloc[-1]) if len(df[df.index.date == d]) > 0 else daily_cash[d]
                         for d in dates}, name="nav")
        nav = nav.sort_index() / total_capital
        return nav

    def get_grid_df(self) -> pd.DataFrame:
        """返回网格线列表用于可视化"""
        if self.state is None:
            return pd.DataFrame()
        rows = []
        for i, (price, bought, sold) in enumerate(self.state.levels):
            status = "已买" if bought else ("已卖" if sold else "待触")
            rows.append({"网格线": f"L{i+1}", "价格": round(price, 4), "状态": status})
        return pd.DataFrame(rows)


def run_grid_backtest(symbol: str, df: pd.DataFrame,
                      grid_type: GridType = "arithmetic",
                      n_levels: int = 10,
                      step_value: float = 0.0,
                      amount_per_grid: float = 10000.0,
                      max_positions: int = 10,
                      initial_capital: float = 0.0,
                      initial_shares: int = 0,
                      commission: float = 0.0003,
                      slippage: float = 0.001) -> tuple[list[Trade], dict, GridEngine]:
    """快捷回测入口

    Args:
        symbol: 标的代码
        df: OHLC DataFrame
        grid_type: 网格类型
        n_levels: 网格线数
        amount_per_grid: 每格金额
        max_positions: 最大持仓格数
        commission: 佣金率
        slippage: 滑点

    Returns:
        (trades, metrics, engine)
    """
    price_low = df["low"].min()
    price_high = df["high"].max()

    config = GridConfig(
        symbol=symbol,
        grid_type=grid_type,
        n_levels=n_levels,
        price_low=price_low,
        price_high=price_high,
        amount_per_grid=amount_per_grid,
        max_positions=max_positions,
        initial_shares=initial_shares,
        initial_capital=initial_capital,
        step_value=step_value,
        commission=commission,
        slippage=slippage,
    )
    engine = GridEngine(config)
    trades = engine.run(df)
    metrics = engine.get_metrics(df)
    return trades, metrics, engine
