import numpy as np
import pandas as pd


def layer_returns(
    factor_signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_layers: int = 5,
) -> dict[int, pd.Series]:
    dates = factor_signals.index.intersection(forward_returns.index)
    layer_portfolios: dict[int, list[float]] = {i: [] for i in range(1, n_layers + 1)}
    layer_dates: list[pd.Timestamp] = []

    for d in dates:
        f = factor_signals.loc[d].dropna()
        r = forward_returns.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_layers * 2:
            continue

        ranked = f.loc[common].rank()
        labels = pd.qcut(ranked, n_layers, labels=False, duplicates="drop")
        if labels is None:
            continue

        labels = labels + 1
        layer_dates.append(d)
        for layer in range(1, n_layers + 1):
            members = labels[labels == layer].index
            if len(members) == 0:
                layer_portfolios[layer].append(np.nan)
            else:
                layer_portfolios[layer].append(r.loc[members].mean())

    result = {}
    for layer, returns in layer_portfolios.items():
        result[layer] = pd.Series(returns, index=pd.DatetimeIndex(layer_dates), name=f"L{layer}")
    return result


def layer_performance(layer_rets: dict[int, pd.Series]) -> pd.DataFrame:
    metrics = []
    for layer, rets in layer_rets.items():
        metrics.append({
            "layer": layer,
            "mean_return": rets.mean(),
            "std": rets.std(),
            "sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0,
            "win_rate": (rets > 0).mean(),
        })
    return pd.DataFrame(metrics).set_index("layer")
