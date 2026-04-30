from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..preprocess.config import AudioPreprocessConfig
from ..preprocess.pipeline import AugmentConfig
from .speech_commands import SpeechCommandsMelDataset, build_official_split_indices


@dataclass(frozen=True)
class BuiltDatasets:
    train: SpeechCommandsMelDataset
    val: SpeechCommandsMelDataset
    test: SpeechCommandsMelDataset


def build_raw_and_augmented(
    speech_commands_root: str | Path,
    cfg: AudioPreprocessConfig,
    *,
    seed: int = 1337,
    device: torch.device | None = None,
) -> tuple[BuiltDatasets, BuiltDatasets]:
    """
    Returns (raw_datasets, augmented_datasets).

    - raw: mel-log + normalization only.
    - augmented: same base pipeline + (time shift + noise + SpecAugment).
    """
    splits = build_official_split_indices(speech_commands_root)

    raw_aug = AugmentConfig(enabled=False)
    train_aug = AugmentConfig(enabled=True)

    raw = BuiltDatasets(
        train=SpeechCommandsMelDataset(splits["train"], cfg, augment=raw_aug, device=device, seed=seed),
        val=SpeechCommandsMelDataset(splits["val"], cfg, augment=raw_aug, device=device, seed=seed),
        test=SpeechCommandsMelDataset(splits["test"], cfg, augment=raw_aug, device=device, seed=seed),
    )
    augmented = BuiltDatasets(
        train=SpeechCommandsMelDataset(splits["train"], cfg, augment=train_aug, device=device, seed=seed),
        val=SpeechCommandsMelDataset(splits["val"], cfg, augment=raw_aug, device=device, seed=seed),
        test=SpeechCommandsMelDataset(splits["test"], cfg, augment=raw_aug, device=device, seed=seed),
    )
    return raw, augmented

