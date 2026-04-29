from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    base_dir: Path

    @property
    def database_path(self) -> Path:
        return self.base_dir / self.raw["paths"]["database"]

    @property
    def reports_dir(self) -> Path:
        return self.base_dir / self.raw["paths"]["reports_dir"]

    @property
    def signals_csv(self) -> Path:
        return self.base_dir / self.raw["paths"]["signals_csv"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(base_dir: str | Path = ".") -> AppConfig:
    base = Path(base_dir).resolve()
    return AppConfig(raw=load_yaml(base / "config.yaml"), base_dir=base)


def load_watchlist(base_dir: str | Path = ".") -> list[dict[str, Any]]:
    base = Path(base_dir).resolve()
    data = load_yaml(base / "watchlist.yaml")
    return [item for item in data.get("watchlist", []) if item.get("enabled", True)]
