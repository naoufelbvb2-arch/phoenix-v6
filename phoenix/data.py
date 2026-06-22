"""
Data utilities for Phoenix v1.

For M0 we use synthetic random-token sequences so there are no HuggingFace
downloads required. The Dataset API is identical to what real data loaders
will use, so swapping in real corpora later is a one-liner.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator


# ---------------------------------------------------------------------------
# Synthetic dataset (M0 / pipeline check)
# ---------------------------------------------------------------------------

class SyntheticTokenDataset(Dataset):
    """
    Generates `n_samples` random token sequences on-the-fly (deterministic
    once the RNG is seeded).
    """
    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int,
        zone_label: str,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.zone_label = zone_label
        rng = torch.Generator()
        rng.manual_seed(seed)
        # pre-generate all tokens so the dataset is fixed
        self.tokens = torch.randint(
            0, vocab_size, (n_samples, seq_len + 1), generator=rng
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        row = self.tokens[idx]
        return {
            "input_ids":  row[:-1],   # (seq_len,)
            "labels":     row[1:],    # (seq_len,)  next-token targets
            "zone_label": self.zone_label,
        }


def zone_collate_fn(batch):
    """
    All samples in a batch must share the same zone_label (enforced by
    ZoneDataLoader below). Returns tensors + a single zone string.
    """
    zone = batch[0]["zone_label"]
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels    = torch.stack([b["labels"]    for b in batch])
    return input_ids, labels, zone


def make_synthetic_loader(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    zone_label: str,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    ds = SyntheticTokenDataset(n_samples, seq_len, vocab_size, zone_label, seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=zone_collate_fn,
        drop_last=True,
    )
