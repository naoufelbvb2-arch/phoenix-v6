"""
Checkpoint save / reload for Phoenix models.

Saved bundle:
  - model state_dict
  - config (as dict)
  - which modules are frozen (list of module paths)
  - training metadata (epoch, step, loss)
"""

import json
import os
import torch
from pathlib import Path
from typing import Any, Dict, List

from .config import PhoenixConfig
from .model import PhoenixModel


def _frozen_modules(model: PhoenixModel) -> List[str]:
    """Return list of named modules where ALL parameters have requires_grad=False."""
    frozen = []
    for name, mod in model.named_modules():
        params = list(mod.parameters(recurse=False))
        if params and all(not p.requires_grad for p in params):
            frozen.append(name)
    return frozen


def save_checkpoint(
    model: PhoenixModel,
    path: str | os.PathLike,
    metadata: Dict[str, Any] | None = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "config": model.cfg.__dict__,
        "frozen_modules": _frozen_modules(model),
        "metadata": metadata or {},
    }
    torch.save(bundle, path)


def load_checkpoint(path: str | os.PathLike, device: str = "cpu") -> PhoenixModel:
    bundle = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = bundle["config"]
    cfg = PhoenixConfig(**cfg_dict)
    model = PhoenixModel(cfg)
    model.load_state_dict(bundle["state_dict"])

    # re-apply frozen flags
    for mod_path in bundle["frozen_modules"]:
        if mod_path == "":
            mod = model
        else:
            mod = model
            for part in mod_path.split("."):
                mod = getattr(mod, part)
        mod.eval()
        for p in mod.parameters(recurse=False):
            p.requires_grad_(False)

    return model


def verify_identical_outputs(
    model_a: PhoenixModel,
    model_b: PhoenixModel,
    tokens: torch.Tensor,
    zone_label: str,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """
    Run both models on the same tokens and return max/mean absolute difference
    in logits. Used by M0 (reload check) and M2 (freeze fidelity).
    """
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        logits_a = model_a(tokens, zone_label)
        logits_b = model_b(tokens, zone_label)
    diff = (logits_a - logits_b).abs()
    result = {
        "max_diff":  diff.max().item(),
        "mean_diff": diff.mean().item(),
        "pass":      diff.max().item() < tol,
        "tol":       tol,
    }
    return result
