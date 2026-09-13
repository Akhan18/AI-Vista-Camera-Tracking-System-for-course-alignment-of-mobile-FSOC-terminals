"""
config_loader.py

Loads AI-VISTA configuration files (YAML) and turns them into a Config
object that the rest of the codebase can use.

WHY THIS EXISTS
----------------
The project instructions require every tunable parameter (world size, camera
FOV, beacon size, noise levels, etc.) to live in a configuration file instead
of being hard-coded inside functions. This module is the single place where
YAML is turned into Python objects, so every other module just does:

    from src.utils.config_loader import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.camera.resolution.width_px)   # 640

DESIGN NOTES (for beginners)
-----------------------------
- A raw dict from `yaml.safe_load()` only supports cfg["camera"]["resolution"],
  which gets noisy fast. We wrap it in a small `Config` class that also
  supports cfg.camera.resolution.width_px (attribute-style access), while
  still behaving like a dict when needed.
- We keep this deliberately simple: no external "config" libraries. A single
  recursive wrapper class is enough for our needs and is easy to read.
- Basic validation is included so that a missing/misspelled key in a config
  file fails loudly and early, instead of causing a confusing crash three
  modules later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    A thin wrapper around a dictionary that allows both:
        cfg["camera"]["resolution"]["width_px"]
    and:
        cfg.camera.resolution.width_px

    This is NOT a general-purpose library — it is intentionally minimal so
    that it stays easy to reason about.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        # Recursively wrap nested dictionaries so attribute access works
        # at every level (e.g. cfg.camera.resolution.width_px).
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain (unwrapped) dictionary form of this config."""
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


# Sections every config file MUST have. If we add new subsystems later
# (e.g. "tracking" specific params), extend this list.
REQUIRED_TOP_LEVEL_KEYS = [
    "world",
    "camera",
    "beacon",
    "motion",
    "disturbances",
    "performance_targets",
    "simulation",
    "paths",
]


def _validate(raw_config: Dict[str, Any], config_path: Path) -> None:
    """
    Check that the loaded YAML has the sections the rest of the system
    depends on. Raises a clear error instead of letting a KeyError happen
    somewhere deep inside the simulation later.
    """
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in raw_config]
    if missing:
        raise ValueError(
            f"Config file '{config_path}' is missing required section(s): "
            f"{missing}. Every AI-VISTA config must define: "
            f"{REQUIRED_TOP_LEVEL_KEYS}"
        )


def load_config(config_path: str | Path, require_full: bool = True) -> Config:
    """
    Load a YAML configuration file and return a Config object.

    Parameters
    ----------
    config_path : str | Path
        Path to a .yaml config file, e.g. "configs/default.yaml"
    require_full : bool, default True
        If True, the file must contain every section listed in
        REQUIRED_TOP_LEVEL_KEYS (this is what we want for a *base* config
        like default.yaml). If False, validation is skipped — use this for
        *scenario override* files (e.g. configs/circular.yaml) that only
        specify the handful of fields they change; they are meant to be
        combined with a base config via merge_configs() and are incomplete
        on their own by design.

    Returns
    -------
    Config
        Attribute-accessible configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML is malformed, or (when require_full=True) missing
        required sections.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{config_path}'. "
            f"Check the path, or run from the project root directory."
        )

    logger.info("Loading config file: %s", config_path)

    with open(config_path, "r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML in '{config_path}': {e}") from e

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Config file '{config_path}' did not parse into a dictionary. "
            f"Check the YAML formatting."
        )

    if require_full:
        _validate(raw_config, config_path)

    logger.info("Config '%s' loaded successfully.", raw_config.get("meta", {}).get(
        "scenario_name", config_path.stem
    ))

    return Config(raw_config)


def load_scenario(base_path: str | Path, scenario_path: str | Path) -> Config:
    """
    Convenience function: load a base config and a partial scenario override,
    and return the merged result. This is the function most application code
    should call.

        cfg = load_scenario("configs/default.yaml", "configs/circular.yaml")

    If scenario_path == base_path, this is equivalent to a plain load_config.
    """
    base = load_config(base_path, require_full=True)
    if Path(scenario_path) == Path(base_path):
        return base
    override = load_config(scenario_path, require_full=False)
    return merge_configs(base, override)


def merge_configs(base: Config, override: Config) -> Config:
    """
    Merge an override config on top of a base config (deep merge).

    This lets scenario files like 'circular.yaml' only specify the fields
    that differ from 'default.yaml', instead of repeating everything.

    Example:
        base = load_config("configs/default.yaml")
        override = load_config("configs/circular.yaml")
        final_cfg = merge_configs(base, override)
    """

    def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_dict)
        for key, value in override_dict.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    merged_dict = _deep_merge(base.to_dict(), override.to_dict())
    return Config(merged_dict)
