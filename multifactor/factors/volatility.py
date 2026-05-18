import numpy as np
import pandas as pd

from .base import Factor


class HistoricalVolatility(Factor):
    name = "historical_volatility"

    def __init__(self, vol_periods: list[int] | None = None, annualize: bool = True):
        self.vol_periods = vol_periods or [20, 60]
        self.annualize = annualize

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change(fill_method=None)
        signals = []
        for period in self.vol_periods:
            vol = ret.rolling(period).std()
            if self.annualize:
                vol = vol * np.sqrt(252)
            signals.append(-vol)
        return sum(signals) / len(signals)


class ParkinsonVolatility(Factor):
    name = "parkinson_volatility"

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        parkinson = (np.log(high / low) ** 2) / (4 * np.log(2))
        vol = parkinson.rolling(self.period).mean().apply(np.sqrt)
        return -vol * np.sqrt(252)


class DownsideRisk(Factor):
    name = "downside_risk"

    def __init__(self, period: int = 60):
        self.period = period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change(fill_method=None)
        downside = ret.where(ret < 0, 0)
        semi_dev = (downside ** 2).rolling(self.period).mean().apply(np.sqrt)
        return -semi_dev * np.sqrt(252)
