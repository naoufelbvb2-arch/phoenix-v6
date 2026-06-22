"""
run_m1.py  -  Phoenix v1 * M1 Training  (Colab-ready)
=======================================================
Set STAGE before running (or pass --stage A/B/EVAL):

  STAGE="A"    Train embed+trunk on mixed corpus; freeze; save backbone_frozen.pt
  STAGE="B"    Load frozen backbone; train Z1 (code) + Z2 (math); save m_base.pt
  STAGE="EVAL" Load m_base.pt; report trunk-only vs zone perplexity per zone

M1 pass criterion: each zone reduces in-domain perplexity >=15% vs trunk-only.

Checkpoints go to Google Drive (or ./checkpoints/m1/ when running locally).
Every epoch is checkpointed. Re-running resumes from the latest epoch automatically.
Stages are independent: A writes backbone_frozen.pt; B reads it. Never re-runs A.

Colab quick-start
-----------------
  !git clone https://github.com/naoufelbvb2-arch/phoenix-v6.git
  %cd phoenix-v6
  !pip install -q datasets transformers

  Then run this script once per stage:
    !python run_m1.py --stage A
    !python run_m1.py --stage B
    !python run_m1.py --stage EVAL

  For a fast CPU debug run (finishes in ~5 min):
    !python run_m1.py --stage A --debug
    !python run_m1.py --stage B --debug
    !python run_m1.py --stage EVAL --debug
"""

# ============================================================
# USER KNOBS  --  edit here or pass as CLI flags
# ============================================================
STAGE = "A"    # "A" | "B" | "EVAL"
DEBUG = False  # True -> tiny data + 1 epoch; finishes on CPU in ~5 min

# Token budgets (real run)
STAGE_A_TOKENS = 4_000_000
Z1_TOKENS      = 3_000_000   # Python code  -> Z1
Z2_TOKENS      = 3_000_000   # Math text    -> Z2
EVAL_TOKENS    = 40_000       # per-zone eval cap

# Hyper-params
STAGE_A_EPOCHS = 3
STAGE_B_EPOCHS = 3
BATCH_SIZE     = 8
SEQ_LEN        = 512
LR             = 3e-4
WEIGHT_DECAY   = 0.1
GRAD_CLIP      = 1.0

# Google Drive base path (ignored outside Colab)
DRIVE_ROOT = "/content/drive/MyDrive/phoenix_v1"
# ============================================================

import argparse, sys, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- CLI overrides -----------------------------------------------------------
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--stage", choices=["A", "B", "EVAL"])
_p.add_argument("--debug", action="store_true")
_a, _ = _p.parse_known_args()
if _a.stage: STAGE = _a.stage
if _a.debug: DEBUG = True

if DEBUG:
    STAGE_A_TOKENS = 10_000
    Z1_TOKENS      = 8_000
    Z2_TOKENS      = 8_000
    EVAL_TOKENS    = 2_000
    STAGE_A_EPOCHS = 1
    STAGE_B_EPOCHS = 1
    BATCH_SIZE     = 4
    SEQ_LEN        = 64

print(f"[config] STAGE={STAGE}  DEBUG={DEBUG}  SEQ_LEN={SEQ_LEN}")

# ============================================================
# DRIVE / CHECKPOINT DIRECTORY
# ============================================================
IN_COLAB = False
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    IN_COLAB = True
except (ImportError, ModuleNotFoundError):
    pass

CKPT_DIR = Path(DRIVE_ROOT) if IN_COLAB else Path("./checkpoints/m1")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[config] Checkpoint dir: {CKPT_DIR}")

# ============================================================
# DEVICE + SEEDS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] Device: {device}")

SEED = 42

def set_seeds(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seeds()

# ============================================================
# TOKENIZER + MODEL CONFIG
# ============================================================
from transformers import AutoTokenizer
from phoenix.config import PhoenixConfig
from phoenix.model import PhoenixModel
from phoenix.train import compute_loss, _reapply_frozen_eval
from phoenix.checkpoint import save_checkpoint, load_checkpoint

print("[init] Loading tokenizer (gpt2)...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
VOCAB_SIZE = tok.vocab_size  # 50257


def make_config() -> PhoenixConfig:
    if DEBUG:
        return PhoenixConfig(
            vocab_size=VOCAB_SIZE, d_model=128, n_heads=4,
            n_trunk_layers=2, trunk_ffn=256, max_seq_len=SEQ_LEN,
            zones={"code": (1, 256), "math": (1, 256)},
        )
    return PhoenixConfig(
        vocab_size=VOCAB_SIZE, d_model=512, n_heads=8,
        n_trunk_layers=6, trunk_ffn=2048, max_seq_len=SEQ_LEN,
        zones={"code": (3, 1536), "math": (3, 1536)},
    )


# ============================================================
# DATA LOADING
# ============================================================
from datasets import load_dataset


def collect_tokens(ds_iter, text_field: str, max_tokens: int) -> torch.Tensor:
    """
    Stream from ds_iter; tokenize with GPT-2 tokenizer; pack into
    shape (N, SEQ_LEN+1) LongTensor.  input_ids = [:, :-1], labels = [:, 1:].
    Stops when N * SEQ_LEN >= max_tokens.
    """
    target_chunks = max(1, max_tokens // SEQ_LEN)
    buf: list[int] = []
    chunks: list[list[int]] = []
    for ex in ds_iter:
        txt = ex.get(text_field, "")
        if not isinstance(txt, str) or not txt.strip():
            continue
        ids = tok.encode(txt, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= SEQ_LEN + 1:
            chunks.append(buf[:SEQ_LEN + 1])
            buf = buf[SEQ_LEN + 1:]
        if len(chunks) >= target_chunks:
            break
    if not chunks:
        raise RuntimeError(
            f"No data collected from field '{text_field}'. "
            "Check dataset name and trust_remote_code."
        )
    return torch.tensor(chunks, dtype=torch.long)


def _cache_path(name: str) -> Path:
    # SEQ_LEN baked into filename so changing it invalidates old caches
    return CKPT_DIR / f"_data_{name}_sl{SEQ_LEN}.pt"


def _cached(name: str, loader_fn, *args):
    p = _cache_path(name)
    if p.exists():
        print(f"  [cache] {p.name}")
        return torch.load(p, weights_only=True)
    data = loader_fn(*args)
    torch.save(data, p)
    print(f"  [cached] {p.name}  ({len(data)} chunks x {SEQ_LEN} tok)")
    return data


def _dl_stage_a() -> torch.Tensor:
    per = max(1, STAGE_A_TOKENS // 3)
    parts = []

    print("    Streaming wikitext-103 (general text)...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1",
                        split="train", streaming=True)
    parts.append(collect_tokens(wiki, "text", per))

    print("    Streaming the-stack-smol / Python (code)...")
    code = load_dataset(
        "bigcode/the-stack-smol", data_dir="data/python",
        split="train", streaming=True, trust_remote_code=True,
    )
    parts.append(collect_tokens(code, "content", per))

    print("    Streaming open-web-math (math)...")
    math_ = load_dataset("open-web-math/open-web-math",
                         split="train", streaming=True)
    parts.append(collect_tokens(math_, "text", per))

    combined = torch.cat(parts)
    g = torch.Generator().manual_seed(SEED)
    return combined[torch.randperm(len(combined), generator=g)]


def _dl_z1() -> torch.Tensor:
    print("    Streaming the-stack-smol / Python (Z1 code)...")
    ds = load_dataset(
        "bigcode/the-stack-smol", data_dir="data/python",
        split="train", streaming=True, trust_remote_code=True,
    )
    return collect_tokens(ds, "content", Z1_TOKENS + EVAL_TOKENS)


def _dl_z2() -> torch.Tensor:
    print("    Streaming open-web-math (Z2 math)...")
    ds = load_dataset("open-web-math/open-web-math",
                      split="train", streaming=True)
    return collect_tokens(ds, "text", Z2_TOKENS + EVAL_TOKENS)


def split_train_eval(data: torch.Tensor, eval_frac: float = 0.1):
    n_eval = max(1, int(len(data) * eval_frac))
    return data[:-n_eval], data[-n_eval:]


def make_loader(data: torch.Tensor, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(data[:, :-1], data[:, 1:])
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        drop_last=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _epoch_path(stage: str, epoch: int) -> Path:
    return CKPT_DIR / f"stage_{stage.lower()}_epoch_{epoch:03d}.pt"


def find_resume(stage: str):
    """Return (path, epoch) for the latest epoch checkpoint, or (None, 0)."""
    pts = sorted(
        CKPT_DIR.glob(f"stage_{stage.lower()}_epoch_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not pts:
        return None, 0
    p = pts[-1]
    return p, int(p.stem.rsplit("_", 1)[-1])


def save_epoch_ckpt(model: PhoenixModel, opt, epoch: int,
                    stage: str, meta: dict | None = None) -> Path:
    p = _epoch_path(stage, epoch)
    torch.save({
        "model_state": model.state_dict(),
        "opt_state":   opt.state_dict(),
        "epoch":       epoch,
        "stage":       stage,
        "config":      model.cfg.__dict__,
        "meta":        meta or {},
    }, p)
    print(f"  [ckpt] {p.name}")
    return p


def load_epoch_ckpt(path: Path, model: PhoenixModel, opt=None):
    b = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(b["model_state"])
    if opt is not None:
        opt.load_state_dict(b["opt_state"])
    return b["epoch"], b.get("meta", {})


def _opt_to_device(opt, dev: torch.device):
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(dev)


# ============================================================
# EVALUATION
# ============================================================

def eval_ppl(model: PhoenixModel, loader: DataLoader,
             zone: str | None) -> float:
    """
    zone=None  -> trunk-only forward (embed -> trunk -> head, no zone).
    zone=name  -> full forward through the named zone.
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = (model.forward_trunk_only(x)
                      if zone is None else model(x, zone))
            total += compute_loss(logits, y).item()
            n += 1
    return math.exp(total / max(n, 1))


def report_specialization(model: PhoenixModel,
                          z1_eval: torch.Tensor,
                          z2_eval: torch.Tensor) -> dict:
    print("\n" + "=" * 62)
    print("M1 Specialization Check  (trunk-only vs trunk+zone PPL)")
    print("=" * 62)
    print(f"  {'Zone':<8} {'Trunk-only':>12} {'Zone PPL':>12} "
          f"{'Reduction':>10} {'Pass?':>7}")
    print("  " + "-" * 55)
    results = {}
    for zone, eval_data in [("code", z1_eval), ("math", z2_eval)]:
        loader  = make_loader(eval_data, shuffle=False)
        ppl_t   = eval_ppl(model, loader, zone=None)
        ppl_z   = eval_ppl(model, loader, zone=zone)
        red_pct = (ppl_t - ppl_z) / ppl_t * 100
        ok      = red_pct >= 15.0
        print(f"  {zone:<8} {ppl_t:>12.2f} {ppl_z:>12.2f} "
              f"{red_pct:>9.1f}% {'PASS' if ok else 'FAIL':>7}")
        results[zone] = {
            "ppl_trunk": ppl_t, "ppl_zone": ppl_z,
            "reduction_pct": red_pct, "pass": ok,
        }
    overall = all(r["pass"] for r in results.values())
    print("=" * 62)
    print(f"M1 overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 62)
    return results


# ============================================================
# STAGE A  --  train embed + trunk on mixed corpus, freeze, save
# ============================================================

def run_stage_a(data_a: torch.Tensor):
    backbone_path = CKPT_DIR / "backbone_frozen.pt"
    if backbone_path.exists():
        print("[Stage A] backbone_frozen.pt already exists -- skipping.")
        return

    cfg   = make_config()
    model = PhoenixModel(cfg).to(device)
    print(f"[Stage A] Total params: {model.count_total():,}")

    # Zones are not trained in Stage A -- freeze them so they never
    # accumulate gradients and are excluded from the optimizer.
    for name in cfg.zone_names():
        model.freeze_zone(name)

    stage_a_params = (list(model.embed.parameters())
                      + list(model.trunk.parameters()))
    opt = torch.optim.AdamW(stage_a_params, lr=LR, weight_decay=WEIGHT_DECAY)

    resume_p, start_ep = find_resume("A")
    if resume_p:
        print(f"[Stage A] Resuming from {resume_p.name}")
        load_epoch_ckpt(resume_p, model, opt)
        model.to(device)
        for name in cfg.zone_names():      # re-apply after load_state_dict
            model.freeze_zone(name)
        _opt_to_device(opt, device)
    else:
        start_ep = 0
        print("[Stage A] Starting from scratch")

    loader    = make_loader(data_a, shuffle=True)
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)

    for ep in range(start_ep + 1, STAGE_A_EPOCHS + 1):
        model.train()
        _reapply_frozen_eval(model)
        losses = []
        t0 = time.time()

        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model.forward_trunk_only(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"  [A] ep={ep} step={step}/{n_batches} "
                      f"loss={loss.item():.4f}")

        avg = sum(losses) / len(losses)
        print(f"[Stage A] ep={ep}  avg_loss={avg:.4f}  "
              f"time={time.time()-t0:.0f}s")
        save_epoch_ckpt(model, opt, ep, "A", {"avg_loss": avg})

    # Freeze backbone permanently, save for Stage B
    model.freeze_backbone()
    torch.save({
        "model_state":    model.state_dict(),
        "config":         cfg.__dict__,
        "frozen_modules": ["embed", "trunk"],
    }, backbone_path)
    print(f"[Stage A] Done.  backbone_frozen.pt saved.")
    print(f"          Everything frozen (embed+trunk+zones).  "
          f"Stage B will unfreeze zones.")


# ============================================================
# STAGE B  --  load backbone, train Z1 + Z2, save M_base
# ============================================================

def run_stage_b(z1_tr, z1_ev, z2_tr, z2_ev):
    mbase_path    = CKPT_DIR / "m_base.pt"
    backbone_path = CKPT_DIR / "backbone_frozen.pt"

    if mbase_path.exists():
        print("[Stage B] m_base.pt already exists -- skipping. "
              "Run STAGE=EVAL to inspect results.")
        return
    if not backbone_path.exists():
        raise FileNotFoundError(
            "backbone_frozen.pt not found. Run STAGE=A first."
        )

    cfg   = make_config()
    model = PhoenixModel(cfg)

    # Load frozen backbone weights then wire up grad flags
    bb = torch.load(backbone_path, map_location="cpu", weights_only=False)
    model.load_state_dict(bb["model_state"])
    model.to(device)
    model.freeze_backbone()
    model.unfreeze_zone("code")
    model.unfreeze_zone("math")
    print(f"[Stage B] Trainable: {model.count_trainable():,} "
          f"/ {model.count_total():,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    resume_p, start_ep = find_resume("B")
    if resume_p:
        print(f"[Stage B] Resuming from {resume_p.name}")
        load_epoch_ckpt(resume_p, model, opt)
        model.to(device)
        model.freeze_backbone()   # re-apply after load_state_dict
        _opt_to_device(opt, device)
    else:
        start_ep = 0
        print("[Stage B] Starting from frozen backbone")

    code_tr = make_loader(z1_tr, shuffle=True)
    math_tr = make_loader(z2_tr, shuffle=True)
    code_ev = make_loader(z1_ev, shuffle=False)
    math_ev = make_loader(z2_ev, shuffle=False)

    for ep in range(start_ep + 1, STAGE_B_EPOCHS + 1):
        model.train()
        _reapply_frozen_eval(model)
        lc: list[float] = []
        lm: list[float] = []
        t0 = time.time()

        # Interleave code and math batches within each epoch.
        # Because routing is explicit, only the active zone's params receive
        # gradients each step -- no cross-zone gradient leakage.
        code_it = iter(code_tr)
        math_it = iter(math_tr)
        done = [False, False]

        while not all(done):
            if not done[0]:
                try:
                    x, y = next(code_it)
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = compute_loss(model(x, "code"), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    lc.append(loss.item())
                except StopIteration:
                    done[0] = True

            if not done[1]:
                try:
                    x, y = next(math_it)
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = compute_loss(model(x, "math"), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    lm.append(loss.item())
                except StopIteration:
                    done[1] = True

        avg_c = sum(lc) / max(len(lc), 1)
        avg_m = sum(lm) / max(len(lm), 1)

        # Per-epoch specialization snapshot
        ppl_ct = eval_ppl(model, code_ev, zone=None)
        ppl_cz = eval_ppl(model, code_ev, zone="code")
        ppl_mt = eval_ppl(model, math_ev, zone=None)
        ppl_mz = eval_ppl(model, math_ev, zone="math")
        red_c  = (ppl_ct - ppl_cz) / ppl_ct * 100
        red_m  = (ppl_mt - ppl_mz) / ppl_mt * 100

        print(f"[Stage B] ep={ep}  "
              f"loss_code={avg_c:.4f}  loss_math={avg_m:.4f}  "
              f"time={time.time()-t0:.0f}s")
        print(f"  code: trunk={ppl_ct:.1f}  zone={ppl_cz:.1f}  "
              f"red={red_c:.1f}%   "
              f"math: trunk={ppl_mt:.1f}  zone={ppl_mz:.1f}  red={red_m:.1f}%")

        save_epoch_ckpt(model, opt, ep, "B", {
            "avg_loss_code": avg_c, "avg_loss_math": avg_m,
            "ppl_code_trunk": ppl_ct, "ppl_code_zone": ppl_cz,
            "ppl_math_trunk": ppl_mt, "ppl_math_zone": ppl_mz,
        })

    save_checkpoint(model, mbase_path, metadata={"stage": "B_complete"})
    print(f"[Stage B] Done.  m_base.pt saved: {mbase_path}")


# ============================================================
# EVAL  --  load M_base, print specialization table
# ============================================================

def run_eval(z1_ev: torch.Tensor, z2_ev: torch.Tensor):
    mbase_path = CKPT_DIR / "m_base.pt"
    if not mbase_path.exists():
        raise FileNotFoundError("m_base.pt not found. Run STAGE=B first.")
    print(f"[EVAL] Loading M_base from {mbase_path}")
    model = load_checkpoint(str(mbase_path), device=str(device))
    model.to(device)
    report_specialization(model, z1_ev, z2_ev)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seeds()
    sfx = "_debug" if DEBUG else ""

    if STAGE == "A":
        print("[data] Loading Stage A mixed corpus...")
        data_a = _cached(f"stage_a{sfx}", _dl_stage_a)
        print(f"[data] {len(data_a)} chunks")
        run_stage_a(data_a)

    elif STAGE == "B":
        print("[data] Loading zone data...")
        z1 = _cached(f"z1{sfx}", _dl_z1)
        z2 = _cached(f"z2{sfx}", _dl_z2)
        z1_tr, z1_ev = split_train_eval(z1)
        z2_tr, z2_ev = split_train_eval(z2)
        print(f"[data] z1 train={len(z1_tr)}  eval={len(z1_ev)}")
        print(f"[data] z2 train={len(z2_tr)}  eval={len(z2_ev)}")
        run_stage_b(z1_tr, z1_ev, z2_tr, z2_ev)

    elif STAGE == "EVAL":
        print("[data] Loading eval slices...")
        z1 = _cached(f"z1{sfx}", _dl_z1)
        z2 = _cached(f"z2{sfx}", _dl_z2)
        _, z1_ev = split_train_eval(z1)
        _, z2_ev = split_train_eval(z2)
        run_eval(z1_ev, z2_ev)

    else:
        raise ValueError(f"Unknown STAGE={STAGE!r}. Use A, B, or EVAL.")


if __name__ == "__main__":
    main()
