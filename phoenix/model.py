"""
Phoenix v1 model.

Hierarchy:
  PhoenixModel
    .embed          : nn.Embedding  (vocab_size x d_model)
    .trunk          : Trunk         (Z5, shared)
    .zones          : nn.ModuleDict  {name -> Zone}
    .head           : tied Linear   (d_model -> vocab_size, weight = embed.weight.T)

  Zone
    .v1             : nn.ModuleList of TransformerBlock
    (later) .v2     : ZoneV2

Adding V2 to a Zone never touches Trunk, other Zones, or embed/head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PhoenixConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> torch.Tensor:
    """Returns (seq_len, head_dim) complex-valued cis tensor."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)          # (seq, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex cis


def _apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """
    x    : (B, n_heads, T, head_dim)   float
    rope : (T, head_dim/2)             complex
    """
    B, H, T, D = x.shape
    x_c = torch.view_as_complex(x.reshape(B, H, T, D // 2, 2).float())
    rope = rope[:T].unsqueeze(0).unsqueeze(0)          # (1, 1, T, D/2)
    x_rot = torch.view_as_real(x_c * rope).flatten(-2) # (B, H, T, D)
    return x_rot.to(x.dtype)


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, use_rope: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # causal mask buffer (not a parameter)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        if self.use_rope and rope is not None:
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)

        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale                   # (B, H, T, T)
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, use_rope: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, max_seq_len, use_rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Trunk (Z5)
# ---------------------------------------------------------------------------

class Trunk(nn.Module):
    def __init__(self, cfg: PhoenixConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.trunk_ffn, cfg.max_seq_len, cfg.rope)
            for _ in range(cfg.n_trunk_layers)
        ])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, rope)
        return x


# ---------------------------------------------------------------------------
# Zone V1
# ---------------------------------------------------------------------------

class ZoneV1(nn.Module):
    """One specialized zone: a stack of transformer blocks."""
    def __init__(self, cfg: PhoenixConfig, n_layers: int, d_internal: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, d_internal, cfg.max_seq_len, cfg.rope)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, rope)
        return x


# ---------------------------------------------------------------------------
# Zone V2  (grown version, Safe-Zero: contributes exactly 0 at init)
# ---------------------------------------------------------------------------

class ZoneV2(nn.Module):
    """
    A grown version attached beneath V1.

    Design:
      - Reads V1's output (the residual stream) -- this IS the bridge; it is
        causal and needs no extra cross-position pooling (which would leak
        future tokens in a causal LM).
      - out_proj is ZERO-INITIALIZED (Safe-Zero) so the contribution is
        EXACTLY 0 at initialization -> the grown model is functionally
        identical to V1-only until V2 is trained.
      - alpha is a learnable scalar gate.

    forward returns the CONTRIBUTION (a delta), which the Zone adds to the
    stream:  x_out = x + (mask *) zoneV2(x)
    """
    def __init__(self, cfg: PhoenixConfig, n_layers: int, d_internal: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, d_internal,
                             cfg.max_seq_len, cfg.rope)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)          # Safe-Zero
        self.alpha = nn.Parameter(torch.tensor(1.0))  # learnable gate

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h, rope)
        return self.alpha * self.out_proj(h)          # delta (0 at init)


# ---------------------------------------------------------------------------
# Zone container (holds V1, optionally V2)
# ---------------------------------------------------------------------------

class Zone(nn.Module):
    """
    Container for zone versions. Holds v1; v2 is attached later via grow().
    Attaching V2 never touches v1, the trunk, other zones, or embed/head.

    v2_mode:
      "off"     -> v1 only (pre-growth behaviour; default)
      "always"  -> x = v1(x) + zoneV2(v1(x))
      "cascade" -> apply V2 only where V1 is uncertain (entropy >= tau)
    """
    def __init__(self, cfg: PhoenixConfig, n_layers: int, d_internal: int):
        super().__init__()
        self.v1 = ZoneV1(cfg, n_layers, d_internal)
        self.v2 = None

    def grow(self, cfg: PhoenixConfig, n_layers: int, d_internal: int):
        self.v2 = ZoneV2(cfg, n_layers, d_internal)

    def forward(self, x: torch.Tensor, rope: torch.Tensor,
                v2_mode: str = "off",
                head_weight: torch.Tensor | None = None,
                tau: float = 0.5) -> torch.Tensor:
        x = self.v1(x, rope)
        if self.v2 is None or v2_mode == "off":
            return x

        contribution = self.v2(x, rope)

        if v2_mode == "always":
            return x + contribution

        if v2_mode == "cascade":
            # entropy of V1's next-token distribution via the (frozen) head
            with torch.no_grad():
                logits = F.linear(x, head_weight)              # (B, T, V)
                p   = F.softmax(logits, dim=-1)
                lp  = F.log_softmax(logits, dim=-1)
                ent = -(p * lp).sum(-1) / math.log(logits.size(-1))  # (B,T) in [0,1]
                mask = (ent >= tau).float().unsqueeze(-1)      # (B,T,1)
            return x + mask * contribution

        raise ValueError(f"unknown v2_mode: {v2_mode}")


# ---------------------------------------------------------------------------
# Phoenix Model
# ---------------------------------------------------------------------------

class PhoenixModel(nn.Module):
    def __init__(self, cfg: PhoenixConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.trunk = Trunk(cfg)
        self.zones = nn.ModuleDict({
            name: Zone(cfg, n_layers, d_internal)
            for name, (n_layers, d_internal) in cfg.zones.items()
        })
        # output head shares weight with embed
        # we do NOT create a separate nn.Linear here to avoid duplicating the
        # parameter; instead we call F.linear at forward time.

        # RoPE cache -- not a parameter, rebuilt on first forward if too short
        self._rope_cache: dict = {}

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed.weight, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in self._rope_cache:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            self._rope_cache[key] = _build_rope_cache(seq_len, head_dim, device)
        return self._rope_cache[key]

    def forward(self, tokens: torch.Tensor, zone_label: str,
                v2_mode: str = "off", tau: float = 0.5) -> torch.Tensor:
        """
        tokens     : (B, T)  long
        zone_label : str, must be in self.zones
        v2_mode    : "off" | "always" | "cascade"   (default "off" = pre-growth)
        returns    : logits (B, T, vocab_size)
        """
        B, T = tokens.shape
        rope = self._get_rope(T, tokens.device)

        x = self.embed(tokens)
        x = self.trunk(x, rope)
        x = self.zones[zone_label](x, rope, v2_mode=v2_mode,
                                   head_weight=self.embed.weight, tau=tau)
        return F.linear(x, self.embed.weight)

    def forward_trunk_only(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed -> trunk -> head, bypassing all zones.
        Used during Stage A training and for the M1 specialization baseline.
        """
        T = tokens.shape[1]
        rope = self._get_rope(T, tokens.device)
        x = self.embed(tokens)
        x = self.trunk(x, rope)
        return F.linear(x, self.embed.weight)

    # -----------------------------------------------------------------------
    # Freeze helpers
    # -----------------------------------------------------------------------

    def freeze_module(self, module: nn.Module):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    def freeze_backbone(self):
        """Freeze embed + trunk after Stage A. Called before Stage B."""
        self.freeze_module(self.embed)
        self.freeze_module(self.trunk)

    def freeze_zone(self, name: str):
        self.freeze_module(self.zones[name])

    def unfreeze_zone(self, name: str):
        self.zones[name].train()
        for p in self.zones[name].parameters():
            p.requires_grad_(True)

    # -----------------------------------------------------------------------
    # Growth helpers (M3 / M4)
    # -----------------------------------------------------------------------

    def grow_zone(self, name: str, n_layers: int | None = None,
                  d_internal: int | None = None):
        """Attach V2 beneath the named zone's V1. Same shape as V1 by default."""
        v1 = self.zones[name].v1
        if n_layers is None:
            n_layers = len(v1.blocks)
        if d_internal is None:
            # FFN up-projection shape: d_model -> d_ff (= d_internal)
            d_internal = v1.blocks[0].ffn.up.out_features
        self.zones[name].grow(self.cfg, n_layers, d_internal)
        print(f"[grow] {name}-V2 attached: {n_layers} layers, d_internal={d_internal}")

    def freeze_all_except_v2(self, name: str):
        """
        Freeze the entire model, then unfreeze only the named zone's V2 + alpha.
        This is the M3/M4 growth-training configuration.
        """
        self.freeze_module(self.embed)
        self.freeze_module(self.trunk)
        for zn in self.zones:
            self.freeze_module(self.zones[zn].v1)
            if self.zones[zn].v2 is not None:
                self.freeze_module(self.zones[zn].v2)
        v2 = self.zones[name].v2
        if v2 is None:
            raise RuntimeError(f"zone '{name}' has no V2 -- call grow_zone first")
        v2.train()
        for p in v2.parameters():
            p.requires_grad_(True)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
