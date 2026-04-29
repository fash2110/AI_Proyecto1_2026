from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CachedItem:
    split: str
    wav_relpath: str
    label_idx: int
    cache_path_raw: str
    cache_path_aug: str
    shape: tuple[int, int, int]


def _parse_item(line: str) -> CachedItem:
    d = json.loads(line)
    return CachedItem(
        split=str(d["split"]),
        wav_relpath=str(d["wav_relpath"]),
        label_idx=int(d["label_idx"]),
        cache_path_raw=str(d["cache_path_raw"]),
        cache_path_aug=str(d.get("cache_path_aug", "")),
        shape=tuple(int(x) for x in d["shape"]),
    )


class CachedSpeechCommandsDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Loads cached mel tensors produced by `precompute_cache.py`.

    - mode='raw': always loads `cache_path_raw`.
    - mode='aug': loads `cache_path_aug` when present (train), else falls back to raw (val/test).
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: Literal["train", "val", "test"],
        mode: Literal["raw", "aug"] = "raw",
        manifest_name: str = "manifest.jsonl",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.split = split
        self.mode = mode

        manifest_path = self.cache_root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        items: list[CachedItem] = []
        for ln in manifest_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            it = _parse_item(ln)
            if it.split == split:
                items.append(it)
        if not items:
            raise ValueError(f"No items found in manifest for split={split!r}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        it = self.items[i]
        rel = it.cache_path_raw
        if self.mode == "aug" and it.cache_path_aug:
            rel = it.cache_path_aug
        path = self.cache_root / rel
        x = torch.load(path, map_location="cpu")
        return x, int(it.label_idx)

