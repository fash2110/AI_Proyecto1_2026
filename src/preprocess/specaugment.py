from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecAugmentConfig:
    # Number of masks per sample
    time_masks: int = 2
    freq_masks: int = 2

    # Max width of masks (in frames / mel bins)
    max_time_mask: int = 20
    max_freq_mask: int = 8

    # Fill strategy
    fill: str = "zero"  # {"zero","mean"}


def spec_augment(
    mel: torch.Tensor, cfg: SpecAugmentConfig, *, rng: torch.Generator
) -> torch.Tensor:
    """
    mel: (n_mels, frames) or (1, n_mels, frames)
    Returns a new tensor with time/freq masking (SpecAugment-style).
    """
    if mel.ndim == 3:
        if mel.shape[0] != 1:
            raise ValueError("Expected channel-first with C=1 for 3D mel.")
        x = mel[0]
        add_channel = True
    elif mel.ndim == 2:
        x = mel
        add_channel = False
    else:
        raise ValueError(f"Expected 2D or 3D mel, got shape {tuple(mel.shape)}")

    n_mels, n_frames = int(x.shape[0]), int(x.shape[1])
    if n_mels == 0 or n_frames == 0:
        return mel

    out = x.clone()
    fill_value = 0.0 if cfg.fill == "zero" else float(out.mean().item())

    def _randint(low: int, high: int) -> int:
        # inclusive low, exclusive high
        return int(torch.randint(low, high, (1,), generator=rng).item())

    # Frequency masks
    for _ in range(cfg.freq_masks):
        w = min(cfg.max_freq_mask, n_mels)
        if w <= 0:
            break
        f = _randint(0, w + 1)
        if f == 0:
            continue
        f0 = _randint(0, max(n_mels - f + 1, 1))
        out[f0 : f0 + f, :] = fill_value

    # Time masks
    for _ in range(cfg.time_masks):
        w = min(cfg.max_time_mask, n_frames)
        if w <= 0:
            break
        t = _randint(0, w + 1)
        if t == 0:
            continue
        t0 = _randint(0, max(n_frames - t + 1, 1))
        out[:, t0 : t0 + t] = fill_value

    if add_channel:
        return out.unsqueeze(0)
    return out

