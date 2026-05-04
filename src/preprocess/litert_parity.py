from __future__ import annotations

from dataclasses import dataclass

from .config import AudioPreprocessConfig


@dataclass(frozen=True)
class LiteRTParityChecklist:
    """
    This is not code that runs in LiteRT; it is a compact spec you must match in mobile.
    Treat it as the single source of truth for the preprocessing contract.
    """

    # Input audio contract
    pcm_encoding: str = "PCM16"
    channels: int = 1
    sample_rate_hz: int = 16_000
    clip_duration_s: float = 1.0
    pad_crop_policy: str = "pad_end_crop_end"

    # STFT contract
    window: str = "hann"
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    center: bool = True
    pad_mode: str = "reflect"

    # Mel contract
    n_mels: int = 64
    f_min_hz: float = 20.0
    f_max_hz: float = 8000.0
    mel_scale: str = "htk"

    # Post
    log_eps: float = 1e-6
    normalization: str = "per_sample_standardize"
    tensor_shape: str = "(1, n_mels, frames)"
    dtype: str = "float32"


def checklist_from_cfg(cfg: AudioPreprocessConfig) -> LiteRTParityChecklist:
    return LiteRTParityChecklist(
        sample_rate_hz=cfg.sample_rate_hz,
        clip_duration_s=cfg.clip_duration_s,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min_hz=float(cfg.f_min_hz),
        f_max_hz=float(cfg.f_max_hz if cfg.f_max_hz is not None else cfg.sample_rate_hz / 2),
        log_eps=float(cfg.eps),
        normalization=cfg.normalize,
    )

