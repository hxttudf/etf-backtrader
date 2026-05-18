import numpy as np
import pandas as pd

from .base import Factor


class Momentum(Factor):
    name = "momentum"

    def __init__(self, roc_periods: list[int] | None = None, skip_days: int = 1, weight_decay: float = 0.5):
        self.roc_periods = roc_periods or [20, 60, 120, 252]
        self.skip_days = skip_days
        self.weight_decay = weight_decay

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        signals = []
        weights = []
        for i, period in enumerate(self.roc_periods):
            roc = close.pct_change(period + self.skip_days, fill_method=None)
            roc_shifted = roc.shift(self.skip_days)
            w = self.weight_decay ** i
            signals.append(roc_shifted * w)
            weights.append(w)

        result = sum(signals) / sum(weights)
        return result


class Momentum52wHigh(Factor):
    name = "momentum_52w_high"

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        rolling_high = close.rolling(self.lookback).max()
        return close / rolling_high - 1.0


class RiskAdjustedMomentum(Factor):
    name = "risk_adjusted_momentum"

    def __init__(self, roc_period: int = 60, vol_period: int = 60):
        self.roc_period = roc_period
        self.vol_period = vol_period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change(fill_method=None)
        roc = close.pct_change(self.roc_period, fill_method=None)
        vol = ret.rolling(self.vol_period).std()
        return roc / vol.replace(0, np.nan).fillna(1e-8)
