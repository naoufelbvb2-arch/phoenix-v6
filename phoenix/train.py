"""
Training loop for Phoenix v1.

A "stage" is a simple object that specifies:
  - which DataLoader(s) to iterate
  - which zone_label(s) they carry
  - how many steps/epochs to run
  - a logging interval

The caller is responsible for setting requires_grad correctly before calling
run_stage(). This keeps the training logic free of stage-specific policy.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional


def compute_loss(
    logits: torch.Tensor,  # (B, T, V)
    labels: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
    )


def run_stage(
    model: nn.Module,
    loaders: List[DataLoader],          # one per zone; batches already zoned
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 10,
    val_fn: Optional[Callable] = None,  # called at end of each epoch
    stage_name: str = "train",
) -> List[float]:
    """
    Train for n_epochs, cycling through all loaders.
    Returns list of per-step losses.
    """
    model.train()
    # frozen submodules must stay in eval mode — re-apply after .train()
    _reapply_frozen_eval(model)

    step = 0
    all_losses = []

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for loader in loaders:
            for input_ids, labels, zone_label in loader:
                input_ids = input_ids.to(device)
                labels    = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, zone_label)
                loss   = compute_loss(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                all_losses.append(loss_val)
                step += 1

                if step % log_interval == 0:
                    print(f"  [{stage_name}] epoch={epoch} step={step}  loss={loss_val:.4f}")

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"[{stage_name}] epoch={epoch} avg_loss={avg:.4f}")

        if val_fn is not None:
            val_fn(epoch)

    return all_losses


def evaluate_perplexity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute perplexity (exp of mean CE loss) over a DataLoader."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for input_ids, labels, zone_label in loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            logits = model(input_ids, zone_label)
            loss   = compute_loss(logits, labels)
            total_loss += loss.item()
            n_batches  += 1
    mean_loss = total_loss / max(n_batches, 1)
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reapply_frozen_eval(model: nn.Module):
    """
    After model.train(), put every module with no-grad params back into .eval()
    so RMSNorm/dropout/etc. stay deterministic. Required because .train() is
    recursive and overrides the eval flag set during freezing.
    """
    for mod in model.modules():
        params = list(mod.parameters(recurse=False))
        if params and all(not p.requires_grad for p in params):
            mod.eval()
