from __future__ import annotations

import math

import torch

from .config import AudioPreprocessConfig


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # HTK-style mel scale
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    f_min_hz: float,
    f_max_hz: float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns mel filterbank matrix of shape (n_mels, n_freq_bins),
    where n_freq_bins = n_fft//2 + 1 for rFFT.
    """
    if f_max_hz is None:
        f_max_hz = sample_rate_hz / 2.0
    if f_min_hz < 0 or f_max_hz <= 0 or f_min_hz >= f_max_hz:
        raise ValueError("Invalid mel frequency range.")

    n_freq_bins = n_fft // 2 + 1

    f_min = torch.tensor(float(f_min_hz), device=device, dtype=torch.float32)
    f_max = torch.tensor(float(f_max_hz), device=device, dtype=torch.float32)

    m_min = _hz_to_mel(f_min)
    m_max = _hz_to_mel(f_max)

    # n_mels triangular filters need n_mels+2 mel points
    m_pts = torch.linspace(m_min, m_max, steps=n_mels + 2, device=device, dtype=torch.float32)
    f_pts = _mel_to_hz(m_pts)

    # Map Hz to FFT bin numbers
    fft_bins = torch.floor((n_fft + 1) * f_pts / sample_rate_hz).to(torch.int64)
    fft_bins = torch.clamp(fft_bins, min=0, max=n_freq_bins - 1)

    fb = torch.zeros((n_mels, n_freq_bins), device=device, dtype=torch.float32)

    for m in range(n_mels):
        left = int(fft_bins[m].item())
        center = int(fft_bins[m + 1].item())
        right = int(fft_bins[m + 2].item())

        if center == left:
            center = min(left + 1, n_freq_bins - 1)
        if right == center:
            right = min(center + 1, n_freq_bins - 1)

        if left < center:
            up = torch.linspace(0.0, 1.0, steps=center - left, device=device, dtype=torch.float32)
            fb[m, left:center] = up
        if center < right:
            down = torch.linspace(1.0, 0.0, steps=right - center, device=device, dtype=torch.float32)
            fb[m, center:right] = torch.maximum(fb[m, center:right], down)

    return fb.to(dtype=dtype)


def mel_log_spectrogram(
    waveform: torch.Tensor,
    cfg: AudioPreprocessConfig,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    waveform: (num_samples,), float32
    returns: (n_mels, num_frames), float32
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(waveform.shape)}")
    if device is None:
        device = waveform.device

    x = waveform.to(device=device, dtype=torch.float32)

    window = torch.hann_window(cfg.win_length, device=device, dtype=torch.float32)
    stft = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    # Power spectrogram: (freq_bins, frames)
    power = (stft.real**2 + stft.imag**2).clamp_min(0.0)

    fb = mel_filterbank(
        sample_rate_hz=cfg.sample_rate_hz,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        f_min_hz=cfg.f_min_hz,
        f_max_hz=cfg.f_max_hz,
        device=device,
        dtype=power.dtype,
    )
    mel = fb @ power  # (n_mels, frames)

    mel = torch.log(mel + cfg.eps)
    return mel


def normalize_mel(mel: torch.Tensor, cfg: AudioPreprocessConfig) -> torch.Tensor:
    """
    mel: (n_mels, frames)
    """
    if cfg.normalize == "none":
        return mel
    if cfg.normalize == "per_sample_standardize":
        mean = mel.mean()
        std = mel.std(unbiased=False).clamp_min(cfg.eps)
        return (mel - mean) / std
    if cfg.normalize == "per_sample_minmax_-1_1":
        mn = mel.min()
        mx = mel.max()
        denom = (mx - mn).clamp_min(cfg.eps)
        mel01 = (mel - mn) / denom
        return mel01 * 2.0 - 1.0
    raise ValueError(f"Unknown normalization: {cfg.normalize}")

