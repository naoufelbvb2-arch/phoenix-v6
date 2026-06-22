from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PhoenixConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_heads: int = 8
    rope: bool = True
    norm: str = "rmsnorm"
    activation: str = "gelu"
    max_seq_len: int = 1024
    tie_embeddings: bool = True

    n_trunk_layers: int = 6
    trunk_ffn: int = 2048

    # zones: name -> (n_layers, d_internal)
    zones: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "code": (3, 1536),
        "math": (3, 1536),
    })

    def zone_names(self):
        return list(self.zones.keys())


def tiny_config() -> PhoenixConfig:
    """M0 pipeline-check config — same code, smaller numbers."""
    return PhoenixConfig(
        vocab_size=8192,
        d_model=256,
        n_heads=4,
        rope=True,
        norm="rmsnorm",
        activation="gelu",
        max_seq_len=256,
        tie_embeddings=True,
        n_trunk_layers=2,
        trunk_ffn=1024,
        zones={"code": (2, 1024)},
    )
