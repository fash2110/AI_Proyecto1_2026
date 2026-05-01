from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .audio import WavData, load_wav_mono_pcm16, prepare_waveform
from .config import AudioPreprocessConfig
from .mel import mel_log_spectrogram, normalize_mel
from .specaugment import SpecAugmentConfig, spec_augment


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = False
    time_shift: bool = True
    max_shift_samples: int = 800
    add_white_noise: bool = True
    snr_db_range: tuple[float, float] = (15.0, 30.0)
    specaugment: bool = True
    specaugment_cfg: SpecAugmentConfig = SpecAugmentConfig()


def waveform_to_input_tensor(
    wav: WavData,
    cfg: AudioPreprocessConfig,
    *,
    augment: AugmentConfig | None = None,
    pad_crop_mode: str = "pad_end_crop_end",
    rng: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns model input tensor shaped (1, n_mels, frames), float32.
    """
    if augment is None:
        augment = AugmentConfig(enabled=False)
    if rng is None:
        rng = torch.Generator(device="cpu")

    x = prepare_waveform(
        wav,
        target_sample_rate_hz=cfg.sample_rate_hz,
        target_num_samples=cfg.num_samples,
        pad_crop_mode=pad_crop_mode,  # type: ignore[arg-type]
        do_time_shift=bool(augment.enabled and augment.time_shift),
        max_shift_samples=int(augment.max_shift_samples),
        do_add_noise=bool(augment.enabled and augment.add_white_noise),
        snr_db_range=augment.snr_db_range,
        rng=rng,
    )

    mel = mel_log_spectrogram(x, cfg, device=device)
    mel = normalize_mel(mel, cfg)
    mel = mel.unsqueeze(0)  # (1, n_mels, frames)

    if augment.enabled and augment.specaugment:
        mel = spec_augment(mel, augment.specaugment_cfg, rng=rng)

    return mel.contiguous()


def wav_path_to_input_tensor(
    wav_path: str | Path,
    cfg: AudioPreprocessConfig,
    *,
    augment: AugmentConfig | None = None,
    rng: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    wav = load_wav_mono_pcm16(wav_path)
    return waveform_to_input_tensor(wav, cfg, augment=augment, rng=rng, device=device)

