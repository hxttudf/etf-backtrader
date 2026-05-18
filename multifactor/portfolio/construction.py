import numpy as np
import pandas as pd


class TopNPortfolio:
    def __init__(
        self,
        top_n: int = 3,
        rebalance_freq: str = "monthly",
        commission: float = 0.0003,
        slippage: float = 0.001,
        min_hold_days: int = 5,
    ):
        self.top_n = top_n
        self.commission = commission
        self.slippage = slippage
        self.min_hold_days = min_hold_days
        self._freq_map = {
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
        }
        self._rebalance_freq = self._freq_map.get(rebalance_freq, "M")

    def rebalance_dates(self, scores: pd.DataFrame) -> pd.DatetimeIndex:
        idx = scores.index
        if len(idx) == 0:
            return idx
        periods = idx.to_series().dt.to_period(self._rebalance_freq)
        keep = ~periods.duplicated(keep="first")
        return idx[keep.values]

    def build_holdings(self, scores: pd.DataFrame) -> pd.DataFrame:
        rebal_dates = self.rebalance_dates(scores)
        holdings = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

        prev_portfolio: list[str] = []
        prev_date: pd.Timestamp | None = None

        for d in rebal_dates:
            if d not in scores.index:
                continue
            day_scores = scores.loc[d].dropna()
            if len(day_scores) < 1:
                continue

            top = day_scores.nlargest(min(self.top_n, len(day_scores))).index.tolist()
            weight = 1.0 / len(top)

            if prev_date is not None and (d - prev_date).days < self.min_hold_days:
                keep = set(prev_portfolio)
                top = list(keep) + [t for t in top if t not in keep]
                top = top[: self.top_n]
                weight = 1.0 / len(top)

            holdings.loc[d, top] = weight
            prev_portfolio = top
            prev_date = d

        holdings = holdings.ffill().fillna(0)
        return holdings


def compute_turnover(holdings: pd.DataFrame) -> pd.Series:
    prev = holdings.shift(1).fillna(0)
    turnover = (holdings - prev).abs().sum(axis=1)
    return turnover


def compute_returns(holdings: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    aligned = holdings.align(asset_returns, join="inner")
    h, r = aligned
    return (h * r).sum(axis=1)
