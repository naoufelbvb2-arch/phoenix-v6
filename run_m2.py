#!/usr/bin/env python3
"""
M2 - Freeze Fidelity Test
==========================
Snapshot Z1 logits, train Z2, snapshot Z1 again. If Z1 is truly frozen the
two snapshots are bit-identical.  Pass: max |A - B| < 1e-4.
"""
import sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from phoenix.model import PhoenixModel
from phoenix.config import PhoenixConfig

DRIVE_ROOT = "/content/drive/MyDrive/phoenix_v1"
THRESHOLD, N_SNAP, N_TRAIN_STEPS, LR, SEQ_LEN, SEED = 1e-4, 5, 100, 1e-4, 512, 42

def get_ckpt_dir():
    try:
        import google.colab  # noqa: F401
        if not Path("/content/drive/MyDrive").exists():
            raise RuntimeError("Mount Drive first in a notebook cell.")
        return Path(DRIVE_ROOT)
    except ImportError:
        return Path("./checkpoints/m1")

def load_m_base(d):
    p = d / "m_base.pt"
    if not p.exists():
        raise FileNotFoundError(f"m_base.pt not found at {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    cfg  = PhoenixConfig(**ckpt["config"])
    m    = PhoenixModel(cfg)
    key  = "state_dict" if "state_dict" in ckpt else "model_state"
    m.load_state_dict(ckpt[key])
    print(f"[M2] Loaded m_base ({sum(p.numel() for p in m.parameters()):,} params)")
    return m

def load_data(d, zone):
    for base in [d, Path("./checkpoints/m1")]:
        p = base / f"_data_{zone}_sl{SEQ_LEN}.pt"
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=False)
            print(f"[M2] {zone}: {len(data)} chunks")
            return data
    raise FileNotFoundError(f"Cache for {zone} not found. Run Stage B first.")

@torch.no_grad()
def snapshot(model, batch, device, label):
    model.eval()
    logits = model(batch.to(device)[:, :-1], zone_label="code").cpu()
    print(f"  [{label}] shape={list(logits.shape)}  max={logits.abs().max():.4f}")
    return logits

def train_z2(model, z2_train, device):
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[M2] Z2 trainable: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=LR)
    model.train(); model.freeze_backbone(); model.freeze_zone("code")
    g = torch.Generator().manual_seed(SEED)
    perm, losses = torch.randperm(len(z2_train), generator=g), []
    for step, idx in enumerate(perm):
        if step >= N_TRAIN_STEPS: break
        x = z2_train[idx].unsqueeze(0).to(device)
        opt.zero_grad()
        logits = model(x[:, :-1], zone_label="math")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        losses.append(loss.item())
        if (step+1) % 25 == 0:
            print(f"  [Z2] step={step+1}/{N_TRAIN_STEPS}  loss={sum(losses[-25:])/25:.4f}")
    print(f"[M2] Z2 done  avg_loss={sum(losses)/len(losses):.4f}")

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = get_ckpt_dir()
    print(f"[M2] Device={device}  Dir={d}")
    model = load_m_base(d).to(device)
    z1 = load_data(d, "z1"); z2 = load_data(d, "z2")
    z1_snap  = z1[-(max(N_SNAP, len(z1)//10)):]
    batch    = torch.stack([z1_snap[i] for i in range(N_SNAP)])
    z2_train = z2[:-(max(1, len(z2)//10))]
    model.freeze_backbone(); model.freeze_zone("code"); model.freeze_zone("math")
    print("\n[M2] -- Snapshot A (before Z2 training) --")
    A = snapshot(model, batch, device, "A")
    print("\n[M2] -- Training Z2 (100 steps) --")
    model.unfreeze_zone("math"); train_z2(model, z2_train, device)
    print("\n[M2] -- Snapshot B (after Z2 training) --")
    B = snapshot(model, batch, device, "B")
    max_d = (A - B).abs().max().item()
    mean_d = (A - B).abs().mean().item()
    print(f"\n[M2] ================================")
    print(f"  max  |A-B| = {max_d:.2e}   threshold = {THRESHOLD:.0e}")
    print(f"  mean |A-B| = {mean_d:.2e}")
    print("==================================")
    if max_d < THRESHOLD:
        print("  OK  M2 PASS -- Z1 bit-stable. Freezing correct. M3 trustworthy.")
    else:
        print("  X   M2 FAIL -- Z1 changed while frozen!")
        print("      Check: shared buffers / RMSNorm stats / dropout in eval")
    return max_d < THRESHOLD

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
