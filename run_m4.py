#!/usr/bin/env python3
"""
M4 - No Cross-Zone Mixing  (family isolation)
==============================================
Question: does growing + training Code-V2 affect the Math family at all?

By explicit routing, a Math sample flows  trunk -> Z2(Math)  and NEVER touches
the Code family or Code-V2. So Math must be BIT-IDENTICAL before vs after a
full Code growth + training cycle.

This is stronger than M3's OLD-preservation: in M3, Python passes THROUGH
Code-V2 (same family), so it can be affected. Here, Math is in a DIFFERENT
family, so it must be untouched by construction.

Protocol:
  1. Load m_base
  2. Snapshot Math logits (route Z2)              -> A   + math ppl_before
  3. Grow Code-V2 and TRAIN it (changes its weights substantially)
  4. Snapshot Math logits again (route Z2)        -> B   + math ppl_after
  5. Compare

Pass: max |A - B| < 1e-4   (effectively bit-identical)
"""

import sys, math
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phoenix.model import PhoenixModel
from phoenix.config import PhoenixConfig

# -- Settings -----------------------------------------------------------------
DRIVE_ROOT  = "/content/drive/MyDrive/phoenix_v1"
SEQ_LEN     = 512
SEED        = 42
THRESHOLD   = 1e-4
SNAP_CHUNKS = 8          # math chunks for the bit-comparison
EVAL_CHUNKS = 40         # math chunks for perplexity
TRAIN_STEPS = 150        # Code-V2 training steps (enough to change its weights)
LR          = 3e-4
BATCH       = 4


def get_ckpt_dir() -> Path:
    try:
        import google.colab  # noqa: F401
        if not Path("/content/drive/MyDrive").exists():
            raise RuntimeError("Mount Drive first in a notebook cell.")
        return Path(DRIVE_ROOT)
    except ImportError:
        return Path("./checkpoints/m1")


def load_m_base(d: Path):
    p = d / "m_base.pt"
    if not p.exists():
        raise FileNotFoundError(f"m_base.pt not found at {p}. Run Stage B first.")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    cfg  = PhoenixConfig(**ckpt["config"])
    m    = PhoenixModel(cfg)
    key  = "state_dict" if "state_dict" in ckpt else "model_state"
    m.load_state_dict(ckpt[key])
    return m


def load_cache(d: Path, name: str):
    for base in [d, Path("./checkpoints/m1")]:
        p = base / f"_data_{name}_sl{SEQ_LEN}.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"_data_{name} not found. Run Stage B first.")


@torch.no_grad()
def snapshot_math(model, batch, device, label):
    """Route a fixed Math batch through Z2; collect logits."""
    model.eval()
    x = batch.to(device)
    logits = model(x[:, :-1], zone_label="math").cpu()   # math has no V2 -> trunk+Z2
    print(f"  [{label}] math logits shape={list(logits.shape)}  "
          f"max={logits.abs().max():.4f}")
    return logits


@torch.no_grad()
def eval_math_ppl(model, chunks, device, n=EVAL_CHUNKS):
    model.eval()
    n = min(n, len(chunks))
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, n, BATCH):
        b = chunks[i:min(i + BATCH, n)].to(device)
        logits = model(b[:, :-1], zone_label="math")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               b[:, 1:].reshape(-1), reduction="sum")
        tot_loss += loss.item(); tot_tok += b[:, 1:].numel()
    return math.exp(tot_loss / tot_tok)


def train_code_v2(model, code_train, device):
    """Grow Code-V2 and train it on code - this MUST NOT affect Math."""
    model.grow_zone("code")
    model.freeze_all_except_v2("code")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[M4] Training Code-V2 ({sum(p.numel() for p in trainable):,} params, "
          f"{TRAIN_STEPS} steps) - Math must stay frozen.")
    opt = torch.optim.AdamW(trainable, lr=LR)

    model.train(); model.freeze_all_except_v2("code")
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(code_train), generator=g)
    losses = []
    for step in range(TRAIN_STEPS):
        idx = perm[(step * BATCH) % (len(perm) - BATCH): (step * BATCH) % (len(perm) - BATCH) + BATCH]
        b = code_train[idx].to(device)
        opt.zero_grad()
        logits = model(b[:, :-1], zone_label="code", v2_mode="always")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  [code-V2] step={step+1}/{TRAIN_STEPS}  "
                  f"loss={sum(losses[-50:])/50:.4f}")


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = get_ckpt_dir()
    print(f"[M4] Device={device}  Dir={d}")

    model = load_m_base(d).to(device)

    math_all = load_cache(d, "z2")
    code_all = load_cache(d, "z1")
    math_snap  = math_all[-(max(SNAP_CHUNKS, len(math_all)//10)):][:SNAP_CHUNKS]
    math_eval  = math_all[-(max(EVAL_CHUNKS, len(math_all)//10)):]
    code_train = code_all[:-(max(1, len(code_all)//10))]

    # -- Snapshot A (before Code growth) --------------------------------------
    print("\n[M4] -- Snapshot A: Math BEFORE Code growth --")
    A = snapshot_math(model, math_snap, device, "A")
    ppl_before = eval_math_ppl(model, math_eval, device)
    print(f"  Math ppl (before) = {ppl_before:.4f}")

    # -- Grow + train Code-V2 -------------------------------------------------
    print("\n[M4] -- Growing + training Code-V2 --")
    train_code_v2(model, code_train, device)

    # -- Snapshot B (after Code growth) ---------------------------------------
    print("\n[M4] -- Snapshot B: Math AFTER Code growth --")
    B = snapshot_math(model, math_snap, device, "B")
    ppl_after = eval_math_ppl(model, math_eval, device)
    print(f"  Math ppl (after)  = {ppl_after:.4f}")

    # -- Compare --------------------------------------------------------------
    diff      = (A - B).abs()
    max_d     = diff.max().item()
    mean_d    = diff.mean().item()
    ppl_drift = abs(ppl_after - ppl_before) / ppl_before

    print(f"\n[M4] ========================================")
    print(f"  max  |A - B| (math logits) = {max_d:.2e}   threshold = {THRESHOLD:.0e}")
    print(f"  mean |A - B|               = {mean_d:.2e}")
    print(f"  Math ppl drift             = {ppl_drift*100:.4f}%")
    print(f"==========================================")

    if max_d < THRESHOLD:
        print("  OK  M4 PASS")
        print("     Math is bit-identical after a full Code growth + training cycle.")
        print("     Explicit routing keeps families isolated - growth in one")
        print("     family cannot leak into another.")
    else:
        print("  FAIL  M4 FAIL - Math changed after Code growth!")
        print("     This should be impossible with explicit routing. Check that")
        print("     math samples are not passing through any Code module, and that")
        print("     the trunk / Z2 are frozen.")

    return max_d < THRESHOLD


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
