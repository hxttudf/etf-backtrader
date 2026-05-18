import numpy as np
import pandas as pd

from .base import Factor


class PriceToMA(Factor):
    name = "price_to_ma"

    def __init__(self, ma_periods: list[int] | None = None):
        self.ma_periods = ma_periods or [20, 60, 200]

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        signals = []
        for period in self.ma_periods:
            ma = close.rolling(period).mean()
            signals.append(close / ma - 1.0)
        return sum(signals) / len(signals)


class MASlope(Factor):
    name = "ma_slope"

    def __init__(self, ma_period: int = 60, slope_period: int = 20):
        self.ma_period = ma_period
        self.slope_period = slope_period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ma = close.rolling(self.ma_period).mean()
        slope = ma.pct_change(self.slope_period, fill_method=None)
        return slope


class ADX(Factor):
    name = "adx"

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        up = high - high.shift(1)
        down = low.shift(1) - low

        plus_dm = pd.DataFrame(
            np.where((up > down) & (up > 0), up.values, 0.0),
            index=up.index, columns=up.columns,
        )
        minus_dm = pd.DataFrame(
            np.where((down > up) & (down > 0), down.values, 0.0),
            index=down.index, columns=down.columns,
        )

        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1, level=0)

        atr = tr.rolling(self.period).mean()
        plus_di = 100 * plus_dm.rolling(self.period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(self.period).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(self.period).mean()
        return adx
