import numpy as np
import pandas as pd

from multifactor.evaluation.ic import rank_ic


class FactorScorer:
    def __init__(
        self,
        method: str = "ic_weighted",
        ic_half_life: int = 60,
        winsorize: float = 3.0,
        neutralization: str = "market",
        rank: bool = True,
    ):
        self.method = method
        self.ic_half_life = ic_half_life
        self.winsorize = winsorize
        self.neutralization = neutralization
        self.rank = rank

    def compute_scores(
        self,
        factor_signals: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        fallback_weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        if self.method == "equal_weighted":
            return self._equal_weighted(factor_signals)
        elif self.method == "ic_weighted":
            return self._ic_weighted(factor_signals, forward_returns, fallback_weights)
        else:
            raise ValueError(f"unknown method: {self.method}")

    def _equal_weighted(self, factor_signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        combined = None
        n = len(factor_signals)
        for name, signals in factor_signals.items():
            z = self._process(signals)
            if combined is None:
                combined = z / n
            else:
                combined += z / n
        return combined

    def _ic_weighted(
        self,
        factor_signals: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        fallback_weights: dict[str, float] | None,
    ) -> pd.DataFrame:
        all_processed = {}
        ic_weights = {}

        for name, signals in factor_signals.items():
            z = self._process(signals)
            all_processed[name] = z

            ic = rank_ic(signals, forward_returns)
            if len(ic) == 0:
                ic_weights[name] = fallback_weights.get(name, 1.0) if fallback_weights else 1.0
            else:
                hl = self.ic_half_life
                decay = 0.5 ** (np.arange(len(ic))[::-1] / hl)
                avg_ic = (ic.values * decay).mean()
                ic_weights[name] = max(0, avg_ic)

        total = sum(ic_weights.values()) or 1.0
        normalized = {k: v / total for k, v in ic_weights.items()}

        combined = None
        for name, z in all_processed.items():
            w = normalized.get(name, 0)
            if combined is None:
                combined = z * w
            else:
                combined += z * w

        return combined

    def _process(self, signals: pd.DataFrame) -> pd.DataFrame:
        signals = signals.copy()
        valid_mask = signals.notna().all(axis=1)
        valid = signals[valid_mask].copy()

        if len(valid) < 2:
            return pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        if self.neutralization != "none":
            if self.neutralization == "market":
                market = valid.mean(axis=1)
                valid = valid.sub(market, axis=0)
        if self.winsorize:
            m = valid.mean(axis=1)
            s = valid.std(axis=1)
            upper = m + self.winsorize * s
            lower = m - self.winsorize * s
            valid = valid.clip(lower, upper, axis=0)
        z = (valid.sub(valid.mean(axis=1), axis=0)).div(
            valid.std(axis=1).replace(0, np.nan), axis=0
        )
        if self.rank:
            z = z.rank(axis=1, pct=True)
        z = z.fillna(0)

        result = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for col in z.columns:
            result.loc[z.index, col] = z[col].values
        return result
