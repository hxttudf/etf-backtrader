"""网格交易引擎 — 东方财富动态基准模型"""

from dataclasses import dataclass
import pandas as pd

T0_PREFIXES = ("15994", "15974", "513", "518", "511", "15998")


@dataclass
class Trade:
    datetime: pd.Timestamp
    side: str
    price: float
    amount: float
    quantity: int
    is_t0: bool = False


@dataclass
class GridConfig:
    symbol: str
    grid_type: str = "arithmetic"
    step_value: float = 0.0
    base_price: float = 0.0
    amount_per_grid: float = 10000.0
    max_positions: int = 10
    initial_capital: float = 0.0
    initial_shares: int = 0
    commission: float = 0.0003
    slippage: float = 0.001


def _is_t0(symbol: str) -> bool:
    return any(symbol.startswith(p) for p in T0_PREFIXES)


class GridEngine:
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.cash: float = 0
        self.position: int = 0
        self.today_bought: int = 0
        self.base_price: float = 0
        self.prev_close: float = 0
        self.trades: list = []
        self._pending_sell: float | None = None
        self._is_t0 = _is_t0(cfg.symbol)

    def _step(self, price):
        if self.cfg.grid_type == "geometric" and self.cfg.step_value > 0:
            return price * self.cfg.step_value / 100
        return self.cfg.step_value if self.cfg.step_value > 0 else price * 0.01

    def _buy_price(self):
        return round(self.base_price - self._step(self.base_price), 4)

    def _sell_price(self):
        return round(self.base_price + self._step(self.base_price), 4)

    def _sellable(self):
        return self.position if self._is_t0 else (self.position - self.today_bought)

    def _process_bar(self, ohlc, dt):
        o, h, l = float(ohlc["open"]), float(ohlc["high"]), float(ohlc["low"])
        c = float(ohlc["close"])
        cfg = self.cfg

        if self.trades and isinstance(dt, pd.Timestamp):
            last_dt = self.trades[-1].datetime
            if isinstance(last_dt, pd.Timestamp) and last_dt.date() < dt.date():
                self.today_bought = 0

        bp = self._buy_price()
        sp = self._sell_price()

        # 卖出：最高价触及卖出价
        if self._pending_sell is not None and sp <= h and self._sellable() > 0:
            qty = min(int(cfg.amount_per_grid / sp), self._sellable())
            if qty > 0:
                rev = qty * sp * (1 - cfg.commission) - 0.1
                self.cash += rev
                self.position -= qty
                self.trades.append(Trade(dt, "sell", sp, rev, qty, self._is_t0))
                self.base_price = sp  # 卖出后基准移到卖出价
                self._pending_sell = None

        # 买入：最低价触及买入价
        if self._pending_sell is None and l <= bp:
            qty = int(cfg.amount_per_grid / (bp * (1 + cfg.slippage)))
            if qty > 0:
                cost = qty * bp * (1 + cfg.commission) + 0.1
                if self.cash >= cost:
                    self.cash -= cost
                    self.position += qty
                    self.today_bought += qty
                    self.trades.append(Trade(dt, "buy", bp, cost, qty, self._is_t0))
                    self.base_price = bp  # 买入后基准移到买入价
                    self._pending_sell = self._sell_price()

        self.prev_close = c

    def run(self, df) -> list:
        if len(df) == 0:
            return []
        self.base_price = self.cfg.base_price if self.cfg.base_price > 0 else float(df["close"].iloc[0])
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else self.cfg.amount_per_grid * self.cfg.max_positions
        init_pos = self.cfg.initial_shares
        self.cash = cap - init_pos * self.base_price
        self.position = init_pos
        self.today_bought = 0
        self.prev_close = 0
        self.trades = []
        self._pending_sell = None
        for dt, row in df.iterrows():
            self._process_bar(row, dt)
        return self.trades

    def get_metrics(self, df) -> dict:
        if not self.trades:
            return {"总收益": 0, "买入次数": 0, "卖出次数": 0}
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else self.cfg.amount_per_grid * self.cfg.max_positions
        fv = self.cash + self.position * float(df["close"].iloc[-1])
        buys = [t for t in self.trades if t.side == "buy"]
        sells = [t for t in self.trades if t.side == "sell"]
        pairs = min(len(buys), len(sells))
        wins = sum(1 for i in range(pairs) if sells[i].price > buys[i].price)
        nav_s = df["close"] * self.position + self.cash
        dd = (nav_s / nav_s.cummax() - 1).min() if len(nav_s) > 0 else 0
        return {"总收益": fv / cap - 1, "买入次数": len(buys), "卖出次数": len(sells),
                "胜率": wins / pairs if pairs > 0 else 0, "最大回撤": dd,
                "初始资金": cap, "最终资产": fv, "持仓份额": self.position, "剩余现金": self.cash}

    def get_nav_series(self, df) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        cap = self.cfg.initial_capital if self.cfg.initial_capital > 0 else self.cfg.amount_per_grid * self.cfg.max_positions
        dates = sorted(set(d.date() for d in df.index))
        pos, cash = 0, cap
        tm = {}
        for t in self.trades:
            d = t.datetime.date()
            tm.setdefault(d, [0, 0.0])
            if t.side == "buy":
                tm[d][0] += t.quantity; tm[d][1] -= t.amount
            else:
                tm[d][0] -= t.quantity; tm[d][1] += t.amount
        nv = {}
        for d in dates:
            if d in tm:
                pos += tm[d][0]; cash += tm[d][1]
            cd = df[df.index.date == d]
            close = float(cd["close"].iloc[-1]) if len(cd) > 0 else 0
            nv[d] = (cash + pos * close) / cap
        return pd.Series(nv, name="nav").sort_index()


def run_grid_backtest(symbol, df, grid_type="arithmetic",
                      step_value=0.0, amount_per_grid=10000.0,
                      max_positions=10, initial_capital=0.0,
                      initial_shares=0, base_price=0.0,
                      commission=0.0003, slippage=0.001):
    config = GridConfig(symbol=symbol, grid_type=grid_type, step_value=step_value,
                        base_price=base_price, amount_per_grid=amount_per_grid,
                        max_positions=max_positions, initial_capital=initial_capital,
                        initial_shares=initial_shares, commission=commission, slippage=slippage)
    engine = GridEngine(config)
    trades = engine.run(df)
    return trades, engine.get_metrics(df), engine
