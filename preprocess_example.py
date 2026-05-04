from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.preprocess.config import AudioPreprocessConfig
from src.preprocess.pipeline import AugmentConfig, wav_path_to_input_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=True, help="Path to a single .wav file")
    ap.add_argument("--aug", action="store_true", help="Enable training-time augmentation")
    args = ap.parse_args()

    cfg = AudioPreprocessConfig()
    augment = AugmentConfig(enabled=bool(args.aug))

    x = wav_path_to_input_tensor(Path(args.wav), cfg, augment=augment, rng=torch.Generator().manual_seed(0))
    print("input tensor shape:", tuple(x.shape), "dtype:", x.dtype, "min/max:", float(x.min()), float(x.max()))


if __name__ == "__main__":
    main()

