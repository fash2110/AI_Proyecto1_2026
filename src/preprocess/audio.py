from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class WavData:
    waveform: torch.Tensor  # (num_samples,), float32 in [-1, 1]
    sample_rate_hz: int


def load_wav_mono_pcm16(path: str | Path) -> WavData:
    """
    Loads a WAV file using stdlib `wave` (PCM16 only) and returns mono float32.

    Speech Commands v0.02 uses 16-bit PCM WAV, which this reader supports.
    """
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        comptype = wf.getcomptype()

        if comptype != "NONE":
            raise ValueError(f"Unsupported WAV compression {comptype!r} for {p}")
        if sampwidth != 2:
            raise ValueError(f"Only PCM16 WAV supported (sampwidth=2). Got {sampwidth} for {p}")

        raw = wf.readframes(n_frames)

    audio_i16 = np.frombuffer(raw, dtype="<i2")  # little-endian int16
    if n_channels > 1:
        audio_i16 = audio_i16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    waveform = torch.from_numpy(audio_i16.astype(np.float32) / 32768.0)
    waveform = waveform.clamp(-1.0, 1.0).contiguous()
    return WavData(waveform=waveform, sample_rate_hz=int(framerate))


def pad_or_crop_1d(
    x: torch.Tensor,
    target_len: int,
    mode: Literal["pad_end_crop_end", "pad_end_crop_center"] = "pad_end_crop_end",
) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(x.shape)}")
    n = int(x.shape[0])
    if n == target_len:
        return x
    if n < target_len:
        pad = target_len - n
        return torch.nn.functional.pad(x, (0, pad))

    # crop
    if mode == "pad_end_crop_end":
        return x[:target_len]
    if mode == "pad_end_crop_center":
        start = max(0, (n - target_len) // 2)
        return x[start : start + target_len]
    raise ValueError(f"Unknown mode: {mode}")


def _linear_resample_1d(x: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    """
    Lightweight resampler (linear interpolation).
    Good enough for this project and replicable in mobile (same formula).
    """
    if src_rate == dst_rate:
        return x
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {tuple(x.shape)}")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive.")

    src_len = int(x.shape[0])
    dst_len = int(round(src_len * (dst_rate / src_rate)))
    if dst_len <= 1 or src_len <= 1:
        return x[:1].repeat(max(dst_len, 1))

    # Create dst time positions in src index units.
    t = torch.linspace(0, src_len - 1, steps=dst_len, device=x.device, dtype=torch.float32)
    t0 = torch.floor(t).to(torch.int64)
    t1 = torch.clamp(t0 + 1, max=src_len - 1)
    w = (t - t0.to(torch.float32)).to(x.dtype)

    x0 = x[t0]
    x1 = x[t1]
    return (1.0 - w) * x0 + w * x1


def time_shift(
    x: torch.Tensor, shift: int, mode: Literal["roll", "pad"] = "roll"
) -> torch.Tensor:
    if shift == 0:
        return x
    if mode == "roll":
        return torch.roll(x, shifts=int(shift), dims=0)
    if mode == "pad":
        shift = int(shift)
        if shift > 0:
            return torch.nn.functional.pad(x, (shift, 0))[: x.shape[0]]
        shift = -shift
        return torch.nn.functional.pad(x, (0, shift))[shift:]
    raise ValueError(f"Unknown mode: {mode}")


def add_white_noise(x: torch.Tensor, snr_db: float, rng: torch.Generator) -> torch.Tensor:
    """
    Adds white noise to reach target SNR in dB (approx).
    """
    if snr_db is None:
        return x
    if not torch.isfinite(x).all():
        return x

    sig_power = x.pow(2).mean().clamp_min(1e-12)
    snr = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype) * torch.sqrt(noise_power)
    return (x + noise).clamp(-1.0, 1.0)


def standardize_waveform(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Optional: normalize waveform amplitude (not the spectrogram) to reduce loudness variance.
    """
    m = x.mean()
    s = x.std(unbiased=False).clamp_min(eps)
    return (x - m) / s


def prepare_waveform(
    wav: WavData,
    target_sample_rate_hz: int,
    target_num_samples: int,
    pad_crop_mode: Literal["pad_end_crop_end", "pad_end_crop_center"] = "pad_end_crop_end",
    *,
    do_time_shift: bool = False,
    max_shift_samples: int = 800,
    do_add_noise: bool = False,
    snr_db_range: tuple[float, float] = (15.0, 30.0),
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns fixed-length waveform at target sample rate.
    Augmentations here are only meant for the *augmented* dataset (training only).
    """
    x = wav.waveform
    x = _linear_resample_1d(x, wav.sample_rate_hz, target_sample_rate_hz)
    x = pad_or_crop_1d(x, target_num_samples, mode=pad_crop_mode)

    if rng is None:
        rng = torch.Generator(device=x.device)

    if do_time_shift:
        shift = int(torch.randint(-max_shift_samples, max_shift_samples + 1, (1,), generator=rng).item())
        x = time_shift(x, shift=shift, mode="roll")

    if do_add_noise:
        low, high = snr_db_range
        snr_db = float((low + (high - low) * torch.rand(1, generator=rng).item()))
        x = add_white_noise(x, snr_db=snr_db, rng=rng)

    return x.contiguous()

