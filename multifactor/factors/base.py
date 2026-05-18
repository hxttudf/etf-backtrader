from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Factor(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ...

    def neutralize(self, signals: pd.DataFrame, method: str = "market") -> pd.DataFrame:
        if method == "none":
            return signals
        if method == "market":
            market = signals.mean(axis=1)
            return signals.sub(market, axis=0)
        if method == "rank":
            return signals.rank(axis=1, pct=True)
        return signals

    def winsorize(self, signals: pd.DataFrame, n: float = 3) -> pd.DataFrame:
        mean = signals.mean(axis=1)
        std = signals.std(axis=1)
        upper = mean + n * std
        lower = mean - n * std
        return signals.clip(lower, upper, axis=0)

    def standardize(self, signals: pd.DataFrame) -> pd.DataFrame:
        mean = signals.mean(axis=1)
        std = signals.std(axis=1)
        return signals.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0).fillna(0)
