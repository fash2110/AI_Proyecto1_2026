from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioPreprocessConfig:
    sample_rate_hz: int = 16_000
    clip_duration_s: float = 1.0

    # STFT
    n_fft: int = 512
    win_length: int = 400  # 25ms at 16kHz
    hop_length: int = 160  # 10ms at 16kHz

    # Mel
    n_mels: int = 64
    f_min_hz: float = 20.0
    f_max_hz: float | None = None  # defaults to sample_rate_hz/2

    # Numerics
    eps: float = 1e-6

    # Normalization
    normalize: str = "per_sample_standardize"  # {"none","per_sample_standardize","per_sample_minmax_-1_1"}

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.clip_duration_s))

