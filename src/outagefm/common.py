from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: str | Path = ROOT / ".env") -> None:
    """Export KEY=VALUE lines of the project .env without overriding variables already set."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        base = read_config(path.parent / parent)
        config = merge(base, config)
    return canonical_config(config)


def merge(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # mkstemp creates mode 0600; give the group the same access as files written with open().
        os.chmod(temporary, 0o660)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment() -> dict:
    versions = {}
    for name in [
        "outagefm",
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "lightgbm",
        "chronos-forecasting",
        "timesfm",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "accelerate",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def provenance(config, split=None):
    return {
        "config": config,
        "environment": environment(),
        "seed": config["seed"],
        "protocol": config["protocol"],
    }


# Module and model-kind aliases for evaluation pickles and checkpoints written under earlier internal names.
LEGACY_MODULES = {"outagefm.foundation_metrics": "outagefm.scoring", "outagefm.baselines": "baseline.models"}
LEGACY_PACKAGE = "outagefield"
LEGACY_KINDS = {"outage_field": "outage_fm"}


def legacy_module(module: str) -> str:
    if module == LEGACY_PACKAGE or module.startswith(LEGACY_PACKAGE + "."):
        module = "outagefm" + module[len(LEGACY_PACKAGE) :]
    return LEGACY_MODULES.get(module, module)


def canonical_kind(kind):
    return LEGACY_KINDS.get(kind, kind)


def canonical_config(config: dict) -> dict:
    """Map an aliased model kind in a loaded config (checkpoint, run record) to its current name."""
    model = config.get("model") if isinstance(config, dict) else None
    if isinstance(model, dict) and model.get("kind") in LEGACY_KINDS:
        config = {**config, "model": {**model, "kind": LEGACY_KINDS[model["kind"]]}}
    return config


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = legacy_module(module)
        if module.startswith("baseline."):
            import_baseline(module)
        return super().find_class(module, name)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return _LegacyUnpickler(stream).load()


def import_baseline(module="baseline.registry"):
    """Baseline implementations live in an optional top-level baseline/ package (not part of this release)."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "baseline":
            raise ModuleNotFoundError("This model kind needs the optional baseline/ package") from error
        raise
