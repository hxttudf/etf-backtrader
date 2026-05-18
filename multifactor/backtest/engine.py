import numpy as np
import pandas as pd

from multifactor.data.loader import DataLoader
from multifactor.data.universe import resolve_universe
from multifactor.portfolio.scoring import FactorScorer
from multifactor.portfolio.construction import TopNPortfolio, compute_turnover, compute_returns
from multifactor.evaluation.ic import rank_ic


class BacktestResult:
    def __init__(
        self,
        holdings: pd.DataFrame,
        portfolio_returns: pd.Series,
        turnover: pd.Series,
        nav: pd.Series,
        factor_scores: pd.DataFrame | None = None,
        factor_ics: dict[str, pd.Series] | None = None,
    ):
        self.holdings = holdings
        self.portfolio_returns = portfolio_returns
        self.turnover = turnover
        self.nav = nav
        self.factor_scores = factor_scores
        self.factor_ics = factor_ics

    @property
    def metrics(self) -> dict:
        rets = self.portfolio_returns.dropna()
        if len(rets) == 0:
            return {}
        total_days = len(rets)
        years = total_days / 252
        cagr = (self.nav.iloc[-1] / self.nav.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        std = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        cummax = self.nav.cummax()
        drawdown = (self.nav - cummax) / cummax
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0
        win_rate = (rets > 0).mean()
        avg_turnover = self.turnover.mean()
        return {
            "cagr": cagr,
            "volatility": std,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "total_days": total_days,
        }


_NEEDS_VOLUME = {"volume_trend", "volume_momentum", "liquidity_screen"}
_NEEDS_OHLC = {"adx", "parkinson_volatility"}


class MultiFactorBacktest:
    def __init__(
        self,
        factors: list,
        factor_weights: dict[str, float] | None = None,
        scorer: FactorScorer | None = None,
        portfolio: TopNPortfolio | None = None,
        data_loader: DataLoader | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 100000.0,
    ):
        self.factors = factors
        self.factor_weights = factor_weights
        self.scorer = scorer or FactorScorer()
        self.portfolio = portfolio or TopNPortfolio()
        self.data_loader = data_loader or DataLoader()
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None
        self.initial_capital = initial_capital

    def _needs_volume(self) -> bool:
        return any(f.name in _NEEDS_VOLUME for f in self.factors)

    def _needs_ohlc(self) -> bool:
        return any(f.name in _NEEDS_OHLC for f in self.factors)

    def run(self, universe: dict[str, str] | list[str] | None = None) -> BacktestResult:
        if isinstance(universe, list):
            u = resolve_universe(universe)
        elif universe is None:
            u = resolve_universe()
        else:
            u = universe

        print(f"[多因子] 加载 {len(u)} 只ETF价格数据...")
        close = self.data_loader.load_extended_prices(u)
        print(f"[多因子] 价格数据: {len(close)}天 ({close.index[0].strftime('%Y-%m-%d')} ~ {close.index[-1].strftime('%Y-%m-%d')})")

        if self.start_date:
            close = close[close.index >= self.start_date]
        if self.end_date:
            close = close[close.index <= self.end_date]

        data = {"close": close}

        if self._needs_volume() or self._needs_ohlc():
            print("[多因子] 加载 OHLC 数据...")
            ohlc_dict = self.data_loader.load_ohlc(u)
            if self._needs_volume():
                print("[多因子] 加载成交量数据...")
                volume = self.data_loader.load_volume(u)
                if volume is not None:
                    data["volume"] = volume.reindex(close.index).ffill()
            if ohlc_dict:
                names = list(u.keys())
                high = pd.DataFrame(
                    {n: ohlc_dict[n]["high"] for n in names if n in ohlc_dict},
                    index=close.index,
                )
                low = pd.DataFrame(
                    {n: ohlc_dict[n]["low"] for n in names if n in ohlc_dict},
                    index=close.index,
                )
                high = high.reindex(close.index).ffill()
                low = low.reindex(close.index).ffill()
                data["high"] = high
                data["low"] = low

        print(f"[多因子] 计算 {len(self.factors)} 个因子...")
        factor_signals = {}
        for f in self.factors:
            try:
                sig = f.compute(data)
                sig = sig.reindex(close.index).ffill()
                factor_signals[f.name] = sig
                print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")

        forward_rets = close.pct_change(fill_method=None).shift(-1)

        print(f"[多因子] IC加权合成评分...")
        scores = self.scorer.compute_scores(factor_signals, forward_rets, self.factor_weights)

        print(f"[多因子] 计算因子IC...")
        factor_ics = {}
        for name, sig in factor_signals.items():
            factor_ics[name] = rank_ic(sig, forward_rets)

        print(f"[多因子] 构建组合 (Top-{self.portfolio.top_n})...")
        holdings = self.portfolio.build_holdings(scores)
        port_rets = compute_returns(holdings, forward_rets)
        turnover = compute_turnover(holdings)

        nav = (1 + port_rets.fillna(0)).cumprod() * self.initial_capital
        print(f"[多因子] 完成! 最终净值: {nav.iloc[-1]:.2f}")

        return BacktestResult(
            holdings=holdings,
            portfolio_returns=port_rets,
            turnover=turnover,
            nav=nav,
            factor_scores=scores,
            factor_ics=factor_ics,
        )
