import numpy as np
import pandas as pd

from .base import Factor


class VolumeTrend(Factor):
    name = "volume_trend"

    def __init__(self, volume_ma_period: int = 20):
        self.volume_ma_period = volume_ma_period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data.get("volume")
        if volume is None:
            raise ValueError("volume data required for VolumeTrend factor")
        vol_ma = volume.rolling(self.volume_ma_period).mean()
        return volume / vol_ma.replace(0, np.nan) - 1.0


class VolumeMomentum(Factor):
    name = "volume_momentum"

    def __init__(self, roc_period: int = 20):
        self.roc_period = roc_period

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data.get("volume")
        if volume is None:
            raise ValueError("volume data required for VolumeMomentum factor")
        return volume.pct_change(self.roc_period, fill_method=None)


class LiquidityScreen(Factor):
    name = "liquidity_screen"

    def __init__(self, min_dollar_volume: float = 1e7):
        self.min_dollar_volume = min_dollar_volume

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data.get("volume")
        if volume is None:
            return pd.DataFrame(1.0, index=close.index, columns=close.columns)
        dollar_vol = close * volume
        return pd.DataFrame(
            np.where(dollar_vol >= self.min_dollar_volume, 1.0, -1.0),
            index=dollar_vol.index,
            columns=dollar_vol.columns,
        )
