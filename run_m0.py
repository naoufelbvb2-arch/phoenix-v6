"""
M0 — Pipeline Sanity Check
===========================
Tiny config (vocab=8192, d_model=256, 2 trunk layers, 1 code zone).
Goal: training loss decreases, no NaNs/crashes, checkpoint saves and
      reloads to bit-identical outputs (max|delta_logit| < 1e-4).
"""

import torch
import random
import numpy as np
from pathlib import Path

from phoenix.config import tiny_config
from phoenix.model import PhoenixModel
from phoenix.data import make_synthetic_loader
from phoenix.train import run_stage, evaluate_perplexity
from phoenix.checkpoint import save_checkpoint, load_checkpoint, verify_identical_outputs

SEED = 42
CKPT_PATH = Path("checkpoints/m0_model.pt")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def main():
    set_seeds(SEED)
    device = torch.device("cpu")
    cfg = tiny_config()

    print("=" * 60)
    print("M0 - Tiny config")
    print(f"  vocab={cfg.vocab_size}  d_model={cfg.d_model}  "
          f"trunk_layers={cfg.n_trunk_layers}  zones={cfg.zones}")

    # -----------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------
    model = PhoenixModel(cfg).to(device)
    total_params = model.count_total()
    print(f"  Total params: {total_params:,}")

    # -----------------------------------------------------------------
    # Synthetic data (zone=code, 512 train samples, 64 val samples)
    # -----------------------------------------------------------------
    train_loader = make_synthetic_loader(
        n_samples=512, seq_len=cfg.max_seq_len, vocab_size=cfg.vocab_size,
        zone_label="code", batch_size=16, seed=SEED, shuffle=True,
    )
    val_loader = make_synthetic_loader(
        n_samples=64, seq_len=cfg.max_seq_len, vocab_size=cfg.vocab_size,
        zone_label="code", batch_size=16, seed=SEED + 1, shuffle=False,
    )

    # -----------------------------------------------------------------
    # Measure initial perplexity (random weights)
    # -----------------------------------------------------------------
    ppl_before = evaluate_perplexity(model, val_loader, device)
    print(f"\n  Perplexity BEFORE training: {ppl_before:.2f}")
    expected_random = cfg.vocab_size  # ~8192 for a random model
    print(f"  (Random-weight baseline ~= vocab_size = {expected_random})")

    # -----------------------------------------------------------------
    # Stage A — train all parameters (no freezing for M0)
    # -----------------------------------------------------------------
    print("\n--- Stage A (train everything, 5 epochs) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    losses = run_stage(
        model=model,
        loaders=[train_loader],
        n_epochs=5,
        optimizer=optimizer,
        device=device,
        log_interval=16,
        stage_name="M0",
    )

    # Check for NaNs
    nan_steps = [i for i, l in enumerate(losses) if not (l == l)]  # NaN != NaN
    if nan_steps:
        print(f"\n  FAIL — NaN detected at steps: {nan_steps[:5]}")
        return

    # -----------------------------------------------------------------
    # Measure perplexity after training
    # -----------------------------------------------------------------
    ppl_after = evaluate_perplexity(model, val_loader, device)
    print(f"\n  Perplexity AFTER training:  {ppl_after:.2f}")

    loss_decreased = losses[-1] < losses[0]
    print(f"  First step loss: {losses[0]:.4f}")
    print(f"  Last  step loss: {losses[-1]:.4f}")
    print(f"  Loss decreased:  {loss_decreased}")

    # -----------------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------------
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, CKPT_PATH, metadata={"epoch": 5, "final_loss": losses[-1]})
    print(f"\n  Checkpoint saved to: {CKPT_PATH}")

    # -----------------------------------------------------------------
    # Reload and verify bit-identical outputs
    # -----------------------------------------------------------------
    model_reload = load_checkpoint(CKPT_PATH, device="cpu")
    model_reload.to(device)

    # Fixed probe batch (same seed every time)
    set_seeds(SEED + 99)
    probe = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len))
    result = verify_identical_outputs(model, model_reload, probe, "code", tol=1e-4)
    print(f"\n  Checkpoint reload check:")
    print(f"    max  |d_logit| = {result['max_diff']:.2e}")
    print(f"    mean |d_logit| = {result['mean_diff']:.2e}")
    print(f"    tol           = {result['tol']:.1e}")
    print(f"    PASS          = {result['pass']}")

    # -----------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    passed = loss_decreased and not nan_steps and result["pass"]
    if passed:
        print("M0 PASS - pipeline is clean.")
    else:
        issues = []
        if not loss_decreased: issues.append("loss did not decrease")
        if nan_steps:          issues.append("NaN detected")
        if not result["pass"]: issues.append(f"checkpoint mismatch (max_diff={result['max_diff']:.2e})")
        print(f"M0 FAIL - issues: {', '.join(issues)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
