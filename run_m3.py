#!/usr/bin/env python3
"""
M3 - Growth + Zero-Forgetting  (the decisive test)
===================================================
Grows Code-V2 on a NEW domain and checks two things at once:
  - OLD preservation : does Python (Code-V1's domain) stay intact?
  - NEW gain         : does V2 actually learn the new domain?

Runs THREE conditions to determine what zero-forgetting requires:
  (i)   always-active, train on NEW only            (freeze-only)
  (ii)  always-active, train on NEW + OLD replay     (freeze + replay)
  (iii) entropy cascade, train on NEW only           (freeze + cascade)

Success per condition:
  OLD perplexity relative INCREASE  < 1%
  NEW perplexity relative DECREASE  >= 10%

Built-in transparency check: right after growth (before any training), the
grown model MUST produce IDENTICAL OLD logits to M_base, because Safe-Zero
makes V2 contribute exactly 0. If that check fails, the growth wiring is wrong.
"""

import sys, math, argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import GPT2TokenizerFast
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from phoenix.model import PhoenixModel
from phoenix.config import PhoenixConfig

# -- Settings -----------------------------------------------------------------
DRIVE_ROOT  = "/content/drive/MyDrive/phoenix_v1"
SEQ_LEN     = 512
SEED        = 42
NEW_TOKENS  = 1_200_000     # new-domain tokens to collect (train + eval)
EVAL_CHUNKS = 40            # chunks used for each perplexity measurement
EPOCHS      = 2
LR          = 3e-4
BATCH       = 4
TAU         = 0.5           # cascade entropy threshold (normalized, [0,1])
ZONE        = "code"        # we grow the code family

OLD_REL_INCREASE_MAX = 0.01    # < 1%
NEW_REL_DECREASE_MIN = 0.10    # >= 10%

tok = GPT2TokenizerFast.from_pretrained("gpt2")


# -- Infra helpers -------------------------------------------------------------
def get_ckpt_dir() -> Path:
    try:
        import google.colab  # noqa: F401
        if not Path("/content/drive/MyDrive").exists():
            raise RuntimeError("Mount Drive first in a notebook cell.")
        return Path(DRIVE_ROOT)
    except ImportError:
        return Path("./checkpoints/m1")


def load_m_base(d: Path):
    """Fresh model from m_base.pt every call (clean isolation per condition)."""
    p = d / "m_base.pt"
    if not p.exists():
        raise FileNotFoundError(f"m_base.pt not found at {p}. Run Stage B first.")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    cfg  = PhoenixConfig(**ckpt["config"])
    m    = PhoenixModel(cfg)
    key  = "state_dict" if "state_dict" in ckpt else "model_state"
    m.load_state_dict(ckpt[key])
    return m


def collect_tokens(stream, field, n_tokens, max_examples=200_000):
    ids = []
    for i, ex in enumerate(stream):
        if i >= max_examples:
            break
        text = ex.get(field) if isinstance(ex, dict) else None
        if not text:
            continue
        ids.extend(tok.encode(text))
        if len(ids) >= n_tokens:
            break
    ids = ids[:n_tokens]
    n_chunks = len(ids) // SEQ_LEN
    ids = ids[: n_chunks * SEQ_LEN]
    return torch.tensor(ids, dtype=torch.long).view(n_chunks, SEQ_LEN)


def load_cache(d: Path, name: str):
    for base in [d, Path("./checkpoints/m1")]:
        p = base / f"_data_{name}_sl{SEQ_LEN}.pt"
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    return None


def load_new_domain(d: Path):
    """
    Try Rust from several sources; fall back to cached MATH data as a
    'new distribution' if all downloads fail. This loader is the ONLY
    swappable part of M3 - the growth mechanism does not depend on it.
    Returns (chunks, label).
    """
    cache = d / f"_data_new_sl{SEQ_LEN}.pt"
    if cache.exists():
        print(f"[M3] new-domain from cache: {cache.name}")
        return torch.load(cache, map_location="cpu", weights_only=False), "rust(cached)"

    attempts = [
        ("bigcode/the-stack-smol-xs", dict(name="rust"), "content"),
        ("codeparrot/github-code",    dict(languages=["Rust"], trust_remote_code=False), "code"),
    ]
    for repo, kwargs, field in attempts:
        try:
            print(f"[M3] trying NEW domain: {repo} {kwargs} ...")
            ds = load_dataset(repo, split="train", streaming=True, **kwargs)
            chunks = collect_tokens(ds, field, NEW_TOKENS)
            if len(chunks) >= EVAL_CHUNKS + 10:
                torch.save(chunks, cache)
                print(f"[M3] NEW = Rust from {repo}  ({len(chunks)} chunks)  cached.")
                return chunks, "rust"
            print(f"[M3]   too few chunks ({len(chunks)}), trying next source.")
        except Exception as e:
            print(f"[M3]   failed: {type(e).__name__}: {str(e)[:120]}")

    # Fallback: cached math data is a genuinely different distribution from Python
    print("[M3] All Rust sources failed -> FALLBACK to cached MATH data as NEW domain.")
    math_chunks = load_cache(d, "z2")
    if math_chunks is None:
        raise RuntimeError("No Rust source and no cached math data. Run Stage B first.")
    return math_chunks, "math(fallback)"


# -- Eval / train --------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, chunks, device, v2_mode="off", tau=TAU, n=EVAL_CHUNKS):
    model.eval()
    n = min(n, len(chunks))
    total_loss, total_tok = 0.0, 0
    for i in range(0, n, BATCH):
        batch = chunks[i:min(i + BATCH, n)].to(device)
        logits = model(batch[:, :-1], zone_label=ZONE, v2_mode=v2_mode, tau=tau)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok  += batch[:, 1:].numel()
    return math.exp(total_loss / total_tok)


def train_v2(model, new_train, device, v2_mode, tau=TAU, replay=None):
    model.freeze_all_except_v2(ZONE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [train] V2 trainable: {sum(p.numel() for p in trainable):,}  mode={v2_mode}")
    opt = torch.optim.AdamW(trainable, lr=LR)

    # Build the training pool (NEW, optionally + OLD replay)
    pool = [("new", new_train)]
    if replay is not None:
        pool.append(("old", replay))

    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        # interleave new + replay chunks
        all_chunks = []
        for _, data in pool:
            idx = torch.randperm(len(data), generator=g)[: len(data)]
            all_chunks.append(data[idx])
        merged = torch.cat(all_chunks, dim=0)
        merged = merged[torch.randperm(len(merged), generator=g)]

        model.train()
        model.freeze_all_except_v2(ZONE)   # re-assert eval mode on frozen parts
        losses = []
        for i in range(0, len(merged), BATCH):
            batch = merged[i:i + BATCH].to(device)
            opt.zero_grad()
            logits = model(batch[:, :-1], zone_label=ZONE, v2_mode=v2_mode, tau=tau)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"  [train] ep={ep+1}/{EPOCHS}  avg_loss={sum(losses)/len(losses):.4f}")


# -- Conditions ----------------------------------------------------------------
def run_condition(label, d, device, old_eval, new_train, new_eval,
                  v2_mode, replay=None):
    print(f"\n{'='*60}\n[M3] CONDITION {label}  (v2_mode={v2_mode}, "
          f"replay={'yes' if replay is not None else 'no'})\n{'='*60}")
    model = load_m_base(d).to(device)
    model.grow_zone(ZONE)                       # attach fresh V2
    train_v2(model, new_train, device, v2_mode, replay=replay)

    old_ppl = eval_ppl(model, old_eval, device, v2_mode=v2_mode)
    new_ppl = eval_ppl(model, new_eval, device, v2_mode=v2_mode)
    return old_ppl, new_ppl


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = get_ckpt_dir()
    print(f"[M3] Device={device}  Dir={d}")

    # -- Data ------------------------------------------------------------------
    python_all = load_cache(d, "z1")
    if python_all is None:
        raise RuntimeError("Python cache (_data_z1) missing. Run Stage B first.")
    new_all, new_label = load_new_domain(d)
    print(f"[M3] OLD=Python ({len(python_all)} chunks)  NEW={new_label} ({len(new_all)} chunks)")

    # held-out evals (last 10%); training on the rest
    n_old_eval = max(EVAL_CHUNKS, len(python_all) // 10)
    old_eval   = python_all[-n_old_eval:]
    n_new_eval = max(EVAL_CHUNKS, len(new_all) // 10)
    new_eval   = new_all[-n_new_eval:]
    new_train  = new_all[:-n_new_eval]
    old_replay = python_all[:-n_old_eval]

    # -- Baselines on M_base (no V2) -------------------------------------------
    print("\n[M3] -- Baselines (M_base, v2 off) --")
    base = load_m_base(d).to(device)
    base_old = eval_ppl(base, old_eval, device, v2_mode="off")
    base_new = eval_ppl(base, new_eval, device, v2_mode="off")
    print(f"  OLD (Python) ppl = {base_old:.3f}")
    print(f"  NEW ({new_label}) ppl = {base_new:.3f}")

    # -- Transparency check: grow, no training -> OLD must be unchanged ---------
    print("\n[M3] -- Transparency check (Safe-Zero) --")
    grown = load_m_base(d).to(device)
    grown.grow_zone(ZONE)
    trans_old = eval_ppl(grown, old_eval, device, v2_mode="always")
    drift = abs(trans_old - base_old) / base_old
    print(f"  OLD ppl with fresh V2 (always) = {trans_old:.6f}  vs base {base_old:.6f}")
    if drift < 1e-4:
        print(f"  OK V2 is transparent at init (drift={drift:.2e}). Growth wiring correct.")
    else:
        print(f"  FAIL V2 NOT transparent (drift={drift:.2e})! Safe-Zero wiring is wrong.")
        print("    Fix before trusting results.")

    # -- Three conditions ------------------------------------------------------
    results = {}
    results["(i) freeze-only"]   = run_condition(
        "(i) freeze-only", d, device, old_eval, new_train, new_eval,
        v2_mode="always", replay=None)
    results["(ii) +replay"]      = run_condition(
        "(ii) +replay", d, device, old_eval, new_train, new_eval,
        v2_mode="always", replay=old_replay)
    results["(iii) cascade"]     = run_condition(
        "(iii) cascade", d, device, old_eval, new_train, new_eval,
        v2_mode="cascade", replay=None)

    # -- Results table ---------------------------------------------------------
    print(f"\n\n{'='*72}")
    print("[M3] RESULTS")
    print(f"  baselines:  OLD(Python)={base_old:.3f}   NEW({new_label})={base_new:.3f}")
    print(f"{'='*72}")
    print(f"{'condition':<18}{'OLD ppl':>10}{'OLD d%':>9}{'NEW ppl':>10}{'NEW d%':>9}  verdict")
    print(f"{'-'*72}")
    for name, (old_ppl, new_ppl) in results.items():
        old_chg = (old_ppl - base_old) / base_old        # + = worse (forgetting)
        new_chg = (new_ppl - base_new) / base_new        # - = better (gain)
        old_ok  = old_chg < OLD_REL_INCREASE_MAX
        new_ok  = (-new_chg) >= NEW_REL_DECREASE_MIN
        verdict = "PASS" if (old_ok and new_ok) else "FAIL"
        flags   = ("" if old_ok else " OLD^") + ("" if new_ok else " NEWv")
        print(f"{name:<18}{old_ppl:>10.3f}{old_chg*100:>+8.2f}%"
              f"{new_ppl:>10.3f}{new_chg*100:>+8.2f}%  {verdict}{flags}")
    print(f"{'='*72}")
    print("  OLD d% should be < +1% (preservation).  NEW d% should be <= -10% (gain).")
    print("  The condition(s) that PASS tell us what zero-forgetting requires:")
    print("    (i) passes  -> freezing alone is enough (simplest).")
    print("    only (ii)   -> replay is needed.")
    print("    only (iii)  -> the entropy cascade is needed.")


if __name__ == "__main__":
    main()
