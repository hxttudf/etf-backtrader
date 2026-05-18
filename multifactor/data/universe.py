from pathlib import Path
import json

_ETF_CONFIG_PATH = Path(__file__).parents[1] / "etf_config.json"
_DEFAULT_GROUPS = {"红纳创黄B": {"红利低波": "512890", "纳指": "159941", "创业板": "159915", "黄金": "159934"}}


def load_etf_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _ETF_CONFIG_PATH
    if not p.exists():
        return _DEFAULT_GROUPS
    with open(p) as f:
        data = json.load(f)
    return data.get("groups", data)


def resolve_universe(group_names: list[str] | None = None, config_path: str | Path | None = None) -> dict[str, str]:
    etf_config = load_etf_config(config_path)
    if group_names is None:
        group_names = list(etf_config.keys())[:1]
    universe: dict[str, str] = {}
    for g in group_names:
        if g in etf_config:
            universe.update(etf_config[g])
    return universe


def group_options(config_path: str | Path | None = None) -> list[str]:
    etf_config = load_etf_config(config_path)
    return list(etf_config.keys())
