from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch.utils.data import Dataset

from ..preprocess.config import AudioPreprocessConfig
from ..preprocess.pipeline import AugmentConfig, wav_path_to_input_tensor
from .splits import resolve_speech_commands_splits, _read_list_file


AUTHORIZED_COMMANDS: tuple[str, ...] = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


@dataclass(frozen=True)
class SpeechCommandsIndex:
    items: list[tuple[Path, int]]  # (wav_path, label_idx)
    label_to_idx: dict[str, int]
    idx_to_label: dict[int, str]


def build_index(root: str | Path, *, commands: Iterable[str] = AUTHORIZED_COMMANDS) -> SpeechCommandsIndex:
    """
    Builds a simple index from the folder structure:
      root/yes/*.wav, root/no/*.wav, ...

    Speech Commands v0.02 is commonly stored exactly this way.
    """
    root = Path(root)
    label_to_idx = {c: i for i, c in enumerate(commands)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    items: list[tuple[Path, int]] = []
    for label, idx in label_to_idx.items():
        folder = root / label
        if not folder.exists():
            continue
        for wav_path in sorted(folder.glob("*.wav")):
            items.append((wav_path, idx))
    if not items:
        raise FileNotFoundError(
            "No WAV files found. Expected Speech Commands layout like root/yes/*.wav etc."
        )
    return SpeechCommandsIndex(items=items, label_to_idx=label_to_idx, idx_to_label=idx_to_label)


class SpeechCommandsMelDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Produces (mel_tensor, label_idx) where mel_tensor is (1, n_mels, frames).
    """

    def __init__(
        self,
        index: SpeechCommandsIndex,
        cfg: AudioPreprocessConfig,
        *,
        augment: AugmentConfig | None = None,
        device: torch.device | None = None,
        seed: int = 1337,
    ) -> None:
        self.index = index
        self.cfg = cfg
        self.augment = augment or AugmentConfig(enabled=False)
        self.device = device
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return len(self.index.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        wav_path, label = self.index.items[i]
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.base_seed + int(i))
        x = wav_path_to_input_tensor(wav_path, self.cfg, augment=self.augment, rng=rng, device=self.device)
        return x, int(label)


def build_official_split_indices(
    root: str | Path,
    *,
    commands: Iterable[str] = AUTHORIZED_COMMANDS,
) -> dict[str, SpeechCommandsIndex]:
    """
    Uses validation_list.txt and testing_list.txt when present.
    Returns dict with keys: train, val, test.
    """
    root = Path(root)
    base_index = build_index(root, commands=commands)

    split_paths = resolve_speech_commands_splits(root)
    val_rel = _read_list_file(split_paths.val_list) if split_paths.val_list else set()
    test_rel = _read_list_file(split_paths.test_list) if split_paths.test_list else set()

    def _rel(p: Path) -> str:
        return str(p.relative_to(root)).replace("\\", "/")

    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    test_items: list[tuple[Path, int]] = []

    for wav_path, y in base_index.items:
        rp = _rel(wav_path)
        if rp in val_rel:
            val_items.append((wav_path, y))
        elif rp in test_rel:
            test_items.append((wav_path, y))
        else:
            train_items.append((wav_path, y))

    def _mk(items: list[tuple[Path, int]]) -> SpeechCommandsIndex:
        return SpeechCommandsIndex(items=items, label_to_idx=base_index.label_to_idx, idx_to_label=base_index.idx_to_label)

    return {"train": _mk(train_items), "val": _mk(val_items), "test": _mk(test_items)}

