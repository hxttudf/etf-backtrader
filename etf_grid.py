"""网格交易引擎 — 参考 QuantDinger 的路径推断 + 信号队列架构"""

from dataclasses import dataclass
from typing import Literal, Optional
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
class ThsGridConfig:
    symbol: str
    price_low: float = 0.0
    price_high: float = 0.0
    base_price: float = 0.0
    trigger_type: Literal["percent", "price"] = "percent"
    buy_threshold: float = 0.01
    sell_threshold: float = 0.01
    bounce_buy: float = 0.0
    pullback_sell: float = 0.0
    amount_per_grid: float = 10000.0
    max_positions: int = 10
    initial_capital: float = 0.0
    initial_shares: int = 0
    commission: float = 0.0003
    stamp_duty: float = 0.0
    slippage: float = 0.001


def _is_t0(symbol: str) -> bool:
    return any(symbol.startswith(p) for p in T0_PREFIXES)


@dataclass
class _GridSignal:
    """网格信号：由策略层生成，交给执行层模拟成交"""
    side: str          # "buy" / "sell"
    price: float       # 触发价（阈值）
    exec_time: pd.Timestamp  # 信号生效时间（该执行的第一根 K 线开盘）
    base_before: float       # 触发前的基准价


class ThsGridEngine:
    def __init__(self, cfg: ThsGridConfig):
        self.cfg = cfg
        self.cash: float = 0.0
        self.position: int = 0
        self.position_count: int = 0
        self.today_bought: int = 0
        self.base_price: float = 0.0
        self.trades: list[Trade] = []
        self._is_t0 = _is_t0(cfg.symbol)
        self._init_grid_positions: int = 0  # 初始底仓占用的网格槽位数，不计入买入容量限制

    def _trigger_prices(self):
        if self.cfg.trigger_type == "percent":
            sell_px = self.base_price * (1 + self.cfg.sell_threshold)
            buy_px = self.base_price * (1 - self.cfg.buy_threshold)
        else:
            sell_px = self.base_price + self.cfg.sell_threshold
            buy_px = self.base_price - self.cfg.buy_threshold
        if self.cfg.price_low > 0:
            buy_px = max(buy_px, self.cfg.price_low)
        if self.cfg.price_high > 0:
            sell_px = min(sell_px, self.cfg.price_high)
        return sell_px, buy_px

    def _sellable(self):
        return self.position if self._is_t0 else (self.position - self.today_bought)

    def _step_qty(self, price):
        return max(1, int(self.cfg.amount_per_grid / price))

    def _can_buy(self):
        return (self.position_count - self._init_grid_positions) < self.cfg.max_positions

    @staticmethod
    def _infer_candle_path(o: float, h: float, l: float, c: float) -> list[float]:
        """推断 K 线内部价格路径（参考 QuantDinger）
        
        看涨 (c >= o): 先下探再拉升  → O → L → H → C
        看跌 (c < o):  先冲高再回落  → O → H → L → C
        
        去重相邻重复价格，避免同一价位触发多次
        """
        raw = [o, l, h, c] if c >= o else [o, h, l, c]
        return [raw[i] for i in range(len(raw)) if i == 0 or abs(raw[i] - raw[i-1]) > 1e-10]

    def _execute_sell(self, price, dt):
        qty = min(self._step_qty(price), self._sellable())
        if qty <= 0 or self.position_count <= 0:
            return
        rev = qty * price * (1 - self.cfg.commission - self.cfg.stamp_duty)
        self.cash += rev
        self.position -= qty
        self.position_count -= 1
        self.base_price = price
        self.trades.append(Trade(dt, "sell", price, rev, qty, self._is_t0))

    def _execute_buy(self, price, dt):
        if not self._can_buy():
            return
        gross_cost = price * (1 + self.cfg.commission)
        qty = self._step_qty(gross_cost)
        if qty <= 0:
            return
        cost = qty * gross_cost
        if self.cash < cost:
            return
        self.cash -= cost
        self.position += qty
        self.position_count += 1
        self.today_bought += qty
        self.base_price = price
        self.trades.append(Trade(dt, "buy", price, cost, qty, self._is_t0))

    # ── 信号生成层（策略框架） ──────────────────────────────

    def _generate_signals(self, df: pd.DataFrame) -> list[_GridSignal]:
        """遍历 K 线，用价格路径推断生成网格信号（含回落/反弹）"""
        signals: list[_GridSignal] = []
        _base = self.base_price
        _pos_cnt = self.position_count
        _today_buy = 0
        _last_date = None
        pull = self.cfg.pullback_sell
        bounce = self.cfg.bounce_buy

        def _can_sell():
            if _pos_cnt <= 0:
                return False
            if self._is_t0:
                return True
            return _pos_cnt - _today_buy > 0

        for dt, row in df.iterrows():
            o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))

            if _last_date is not None and _last_date != dt.date():
                _today_buy = 0
            _last_date = dt.date()

            if (self.cfg.price_low > 0 and h < self.cfg.price_low) or (self.cfg.price_high > 0 and l > self.cfg.price_high):
                continue

            sell_px = _base * (1 + self.cfg.sell_threshold) if self.cfg.trigger_type == "percent" else _base + self.cfg.sell_threshold
            buy_px = _base * (1 - self.cfg.buy_threshold) if self.cfg.trigger_type == "percent" else _base - self.cfg.buy_threshold
            if self.cfg.price_low > 0:
                buy_px = max(buy_px, self.cfg.price_low)
            if self.cfg.price_high > 0:
                sell_px = min(sell_px, self.cfg.price_high)

            path = self._infer_candle_path(o, h, l, c)
            i = 0
            while i < len(path):
                px = path[i]
                acted = False

                init_gp = self._init_grid_positions

                # 检查卖出（含回落）
                if px >= sell_px and _can_sell():
                    if pull > 0:
                        peak = px
                        for j in range(i + 1, len(path)):
                            peak = max(peak, path[j])
                            pull_px = peak * (1 - pull)
                            if path[j] <= pull_px:
                                signals.append(_GridSignal("sell", pull_px, dt, _base))
                                _base = pull_px
                                _pos_cnt -= 1
                                acted = True
                                i = j
                                break
                        if not acted:
                            signals.append(_GridSignal("sell", sell_px, dt, _base))
                            _base = sell_px
                            _pos_cnt -= 1
                            acted = True
                    else:
                        signals.append(_GridSignal("sell", sell_px, dt, _base))
                        _base = sell_px
                        _pos_cnt -= 1
                        acted = True

                # 检查买入（含反弹）
                elif px <= buy_px and (_pos_cnt - init_gp) < self.cfg.max_positions:
                    if bounce > 0:
                        valley = px
                        for j in range(i + 1, len(path)):
                            valley = min(valley, path[j])
                            bounce_px = valley * (1 + bounce)
                            if path[j] >= bounce_px:
                                signals.append(_GridSignal("buy", bounce_px, dt, _base))
                                _base = bounce_px
                                _pos_cnt += 1
                                _today_buy += 1
                                acted = True
                                i = j
                                break
                        if not acted:
                            signals.append(_GridSignal("buy", buy_px, dt, _base))
                            _base = buy_px
                            _pos_cnt += 1
                            _today_buy += 1
                            acted = True
                    else:
                        signals.append(_GridSignal("buy", buy_px, dt, _base))
                        _base = buy_px
                        _pos_cnt += 1
                        _today_buy += 1
                        acted = True

                if acted:
                    sell_px = _base * (1 + self.cfg.sell_threshold) if self.cfg.trigger_type == "percent" else _base + self.cfg.sell_threshold
                    buy_px = _base * (1 - self.cfg.buy_threshold) if self.cfg.trigger_type == "percent" else _base - self.cfg.buy_threshold
                i += 1

        return signals

    # ── 执行层 ────────────────────────────────────────

    def _execute_signals(self, signals: list[_GridSignal], df_minute: Optional[pd.DataFrame]):
        """逐条执行信号，如有分钟线则精确定位成交时间"""
        _last_sig_date = None
        for sig in signals:
            # T+1 日切
            if _last_sig_date is not None and _last_sig_date != sig.exec_time.date():
                self.today_bought = 0
            _last_sig_date = sig.exec_time.date()

            self.base_price = sig.base_before
            sell_px, buy_px = self._trigger_prices()
            exec_price = sig.price

            if df_minute is not None:
                # 多时间框架：在分钟线中寻找精确成交点
                day = sig.exec_time.normalize()
                day_end = day + pd.Timedelta(days=1)
                day_bars = df_minute[(df_minute.index >= day) & (df_minute.index < day_end)]
                found = False
                for dt_m, row_m in day_bars.iterrows():
                    om, hm, lm, cm = (float(row_m[k]) for k in ("open", "high", "low", "close"))
                    for px in self._infer_candle_path(om, hm, lm, cm):
                        if sig.side == "sell" and px >= exec_price:
                            self._execute_sell(exec_price, dt_m)
                            found = True
                            break
                        if sig.side == "buy" and px <= exec_price:
                            self._execute_buy(exec_price, dt_m)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    if len(day_bars) > 0:
                        # 用当日最后一根 K 线的收盘价执行
                        last = day_bars.iloc[-1]
                        last_c = float(last["close"])
                        if sig.side == "sell":
                            exec_px = min(sig.price, last_c)
                            self._execute_sell(exec_px, sig.exec_time)
                        else:
                            exec_px = max(sig.price, last_c)
                            self._execute_buy(exec_px, sig.exec_time)
                    else:
                        # 当日没有分钟线数据，回退到日线执行
                        if sig.side == "sell":
                            self._execute_sell(exec_price, sig.exec_time)
                        else:
                            self._execute_buy(exec_price, sig.exec_time)
            else:
                # 单时间框架：直接用信号时间执行
                if sig.side == "sell":
                    self._execute_sell(exec_price, sig.exec_time)
                else:
                    self._execute_buy(exec_price, sig.exec_time)

    def run(self, df: pd.DataFrame, df_minute: Optional[pd.DataFrame] = None) -> list[Trade]:
        """运行回测
        
        Args:
            df: 策略时间框架 K 线 (日线 / 信号频率)
            df_minute: 可选，分钟线用于精确成交
        """
        if len(df) == 0:
            return []
        self.base_price = self.cfg.base_price if self.cfg.base_price > 0 else float(df["close"].iloc[0])
        trading_cap = self.cfg.amount_per_grid * self.cfg.max_positions
        init_shares = int(self.cfg.initial_shares / self.base_price) if self.cfg.initial_shares > 0 else 0
        step_shares = self._step_qty(self.base_price)
        init_pos = int(init_shares / step_shares) if step_shares > 0 else 0
        cap = self.cfg.initial_shares + trading_cap
        self._init_cash = trading_cap
        self._init_pos = init_shares
        self.cash = trading_cap
        self.position = init_shares
        self._init_grid_positions = min(init_pos, self.cfg.max_positions)
        self.position_count = self._init_grid_positions
        self.today_bought = 0
        self.trades = []

        # Phase 1: 生成信号（策略层）
        signals = self._generate_signals(df)

        # Phase 2: 重置状态，执行信号（执行层）
        self.base_price = self.cfg.base_price if self.cfg.base_price > 0 else float(df["close"].iloc[0])
        self._init_cash = trading_cap
        self._init_pos = init_shares
        self.cash = trading_cap
        self.position = init_shares
        self._init_grid_positions = min(init_pos, self.cfg.max_positions)
        self.position_count = self._init_grid_positions
        self.today_bought = 0
        self.trades = []

        self._execute_signals(signals, df_minute)
        return self.trades

    def get_metrics(self, df: pd.DataFrame) -> dict:
        cap = self.cfg.initial_shares + self.cfg.amount_per_grid * self.cfg.max_positions
        first_close = float(df["close"].iloc[0])
        last_close = float(df["close"].iloc[-1])
        fv = self.cash + self.position * last_close
        buys = [t for t in self.trades if t.side == "buy"]
        sells = [t for t in self.trades if t.side == "sell"]
        pairs = min(len(buys), len(sells))
        wins = sum(1 for i in range(pairs) if sells[i].price > buys[i].price)
        nav_s = df["close"] * self.position + self.cash
        dd = (nav_s / nav_s.cummax() - 1).min() if len(nav_s) > 0 and nav_s.cummax().iloc[-1] > 0 else 0
        grid_ret = fv / cap - 1 if cap > 0 else 0
        hold_ret = last_close / first_close - 1 if first_close > 0 else 0
        return {"总收益": grid_ret, "持有不动收益": hold_ret, "网格超额收益": grid_ret - hold_ret,
                "买入次数": len(buys), "卖出次数": len(sells), "胜率": wins / pairs if pairs > 0 else 0,
                "最大回撤": dd,
                "初始资金": cap, "最终资产": fv, "持仓份额": self.position, "剩余现金": self.cash}

    def get_nav_series(self, df: pd.DataFrame) -> pd.Series:
        cap = self.cfg.initial_shares + self.cfg.amount_per_grid * self.cfg.max_positions
        dates = sorted(set(d.date() for d in df.index))
        pos = getattr(self, '_init_pos', 0)
        cash = getattr(self, '_init_cash', cap)
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
            nv[d] = (cash + pos * close) / cap if cap > 0 else 1.0
        return pd.Series(nv, name="nav").sort_index()


def run_grid_backtest(symbol: str, df: pd.DataFrame,
                      price_low: float = 0.0, price_high: float = 0.0,
                      base_price: float = 0.0,
                      trigger_type: str = "percent",
                      buy_threshold: float = 0.01, sell_threshold: float = 0.01,
                      bounce_buy: float = 0.0, pullback_sell: float = 0.0,
                      amount_per_grid: float = 10000.0,
                      max_positions: int = 10, initial_capital: float = 0.0,
                      initial_shares: int = 0,
                      commission: float = 0.0003, stamp_duty: float = 0.0,
                      slippage: float = 0.001,
                      df_minute: Optional[pd.DataFrame] = None):
    config = ThsGridConfig(symbol=symbol, price_low=price_low, price_high=price_high,
                           base_price=base_price,
                           trigger_type=trigger_type,
                           buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                           bounce_buy=bounce_buy, pullback_sell=pullback_sell,
                           amount_per_grid=amount_per_grid,
                           max_positions=max_positions, initial_capital=initial_capital,
                           initial_shares=initial_shares,
                           commission=commission, stamp_duty=stamp_duty, slippage=slippage)
    engine = ThsGridEngine(config)
    trades = engine.run(df, df_minute)
    return trades, engine.get_metrics(df), engine
