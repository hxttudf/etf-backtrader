import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rank_ic(factor_signals: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    dates = factor_signals.index.intersection(forward_returns.index)
    ic_values = {}
    for d in dates:
        f = factor_signals.loc[d].dropna()
        r = forward_returns.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 3:
            continue
        ic, _ = spearmanr(f.loc[common], r.loc[common])
        ic_values[d] = ic
    return pd.Series(ic_values, name="rank_ic")


def icir(ic_series: pd.Series) -> float:
    return ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0.0


def cumulative_ic(ic_series: pd.Series) -> pd.Series:
    return ic_series.cumsum()


def ic_summary(factor_signals: pd.DataFrame, forward_returns: pd.DataFrame) -> dict:
    ic = rank_ic(factor_signals, forward_returns)
    return {
        "mean_ic": ic.mean(),
        "std_ic": ic.std(),
        "icir": icir(ic),
        "hit_rate": (ic > 0).mean(),
        "cumulative_ic": cumulative_ic(ic),
    }
