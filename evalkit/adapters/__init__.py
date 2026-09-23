"""
Adapter registry.

Any .py file dropped in this folder that defines a DetectorAdapter subclass is
picked up automatically — the filename becomes the --format value. No
registration step, no editing this file.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

from .base import EMPTY, DetectorAdapter, letterbox, nms_numpy, scale_boxes

_DIR = Path(__file__).parent
_REGISTRY: dict[str, type] = {}


def _discover() -> None:
    for py in sorted(_DIR.glob("*.py")):
        if py.stem in ("__init__", "base"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{py.stem}")
        except Exception as exc:  # noqa: BLE001
            print(f"[evalkit] skipped adapter '{py.stem}': {exc}")
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, DetectorAdapter)
                and obj is not DetectorAdapter
                and obj.__module__ == mod.__name__
            ):
                # filename wins, so my_model.py -> --format my_model
                key = py.stem.replace("_adapter", "")
                _REGISTRY[key] = obj


_discover()


def list_formats() -> list[str]:
    return sorted(_REGISTRY)


def load_external(path: str | Path) -> type:
    """
    Import a predictor that lives OUTSIDE evalkit — in the model author's own
    repo, next to their own weights and their own dependencies.

        python run_eval.py --predictor ~/my_detector/evalkit_predictor.py ...

    This is the escape hatch that means nobody edits evalkit to add a model.
    It also sidesteps the real blocker: two teammates' models often need
    incompatible packages (this very repo cannot have `ultralytics` installed —
    see requirements.txt), so a single shared environment that imports every
    model is not achievable. Each author runs from their own environment.
    """
    import importlib.util

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No predictor file at {p}")

    # The predictor almost always imports from its own repo, so make that repo
    # importable the way `python my_repo/predictor.py` would.
    repo = str(p.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    spec = importlib.util.spec_from_file_location(f"evalkit_external_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {p} as a Python module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    found = [
        obj
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, DetectorAdapter)
        and obj is not DetectorAdapter
        and obj.__module__ == mod.__name__
    ]
    if not found:
        raise ImportError(
            f"{p} defines no DetectorAdapter subclass.\n"
            f"It must contain a class inheriting DetectorAdapter with load() and "
            f"predict(). Start from evalkit/templates/external_predictor.py."
        )
    if len(found) > 1:
        raise ImportError(
            f"{p} defines {len(found)} DetectorAdapter subclasses "
            f"({', '.join(c.__name__ for c in found)}); keep exactly one."
        )
    return found[0]


def get_adapter_class(fmt: str) -> type:
    if fmt not in _REGISTRY:
        raise ValueError(
            f"Unknown format '{fmt}'. Available: {list_formats()}. "
            f"For a non-standard model, copy adapters/custom_template.py."
        )
    return _REGISTRY[fmt]


def get_adapter(fmt: str, **kwargs) -> DetectorAdapter:
    return get_adapter_class(fmt)(**kwargs)


def detect_format(weights: str | Path) -> str:
    """
    Guess the format from the file itself so --format is usually optional.
    Returns a key from list_formats(), or raises with a clear message.
    """
    p = Path(weights)
    suffix = p.suffix.lower()

    if suffix == ".onnx":
        return "onnx"
    if suffix in (".engine", ".plan", ".trt"):
        return "tensorrt"
    if suffix == ".torchscript":
        return "torchscript"

    if suffix in (".pt", ".pth"):
        try:
            import torch
        except ImportError:
            raise RuntimeError("pip install torch to auto-detect .pt files") from None

        # TorchScript archives fail torch.load but load with jit.
        try:
            torch.jit.load(str(p), map_location="cpu")
            return "torchscript"
        except Exception:  # noqa: BLE001
            pass

        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            inner = ckpt["model"]
            if not hasattr(inner, "state_dict"):
                return "visionkit"  # plain state_dict, needs a plugin to rebuild
            # A live nn.Module — but whose? Ultralytics and this repo both pickle
            # one, and they are not interchangeable: loading a repo-trained
            # DetectionModel through ultralytics.YOLO() either fails to unpickle
            # or feeds 3 channels to a 1-channel grayscale net. Tell them apart
            # by where the class was defined.
            module = type(inner).__module__ or ""
            if module.startswith("ultralytics"):
                return "ultralytics"
            if module.startswith("models."):  # this repo's models/yolo.py
                return "yolov5"
            return "ultralytics"
        raise RuntimeError(
            f"Could not identify '{p.name}'. Pass --format explicitly, or copy "
            f"adapters/custom_template.py if it's a custom architecture."
        )

    raise RuntimeError(f"Unrecognised weights extension '{suffix}'. Pass --format.")


__all__ = [
    "EMPTY",
    "DetectorAdapter",
    "detect_format",
    "get_adapter",
    "get_adapter_class",
    "load_external",
    "letterbox",
    "list_formats",
    "nms_numpy",
    "scale_boxes",
]
