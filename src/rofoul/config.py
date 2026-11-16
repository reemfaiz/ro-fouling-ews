"""Configuration loading: packaged defaults deep-merged with a user YAML file."""
from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

ENV_CONFIG = "ROFOUL_CONFIG"
ENV_MODEL = "ROFOUL_MODEL_PATH"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def default_config_path() -> Path:
    """The default.yaml shipped inside the package (kept in sync with config/)."""
    return Path(str(resources.files("rofoul") / "default.yaml"))


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(default_config_path(), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    path = path or os.environ.get(ENV_CONFIG)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Config file {p} must contain a YAML mapping")
        cfg = _deep_merge(cfg, user)
        cfg["_config_path"] = str(p.resolve())
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    if os.environ.get(ENV_MODEL):
        cfg["model"]["path"] = os.environ[ENV_MODEL]
    return cfg


def resolve_model_path(cfg: dict) -> Path:
    """Resolve model.path against CWD, then the config file's dir, then the repo root."""
    raw = Path(cfg["model"]["path"]).expanduser()
    if raw.is_absolute():
        return raw
    candidates = [Path.cwd() / raw]
    if cfg.get("_config_path"):
        cp = Path(cfg["_config_path"]).parent
        candidates += [cp / raw, cp.parent / raw]
    candidates.append(Path(__file__).resolve().parents[2] / raw)  # repo root (editable install)
    candidates.append(Path("/app") / raw)  # docker image layout
    for c in candidates:
        if (c / "model.joblib").exists():
            return c
    return candidates[0]
