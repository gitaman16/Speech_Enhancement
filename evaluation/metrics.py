"""
Speech quality metrics.

Three standard metrics used to measure how well the model enhances speech:

STOI (Short-Time Objective Intelligibility)
    Range: 0 to 1. Measures how intelligible the speech is.
    Higher is better. Values above 0.85 are considered good.

PESQ (Perceptual Evaluation of Speech Quality)
    Range: -0.5 to 4.5 (wideband). Approximates human quality judgements.
    Higher is better. Telephone quality ≈ 3.5, broadcast quality ≈ 4.5.

SNR (Signal-to-Noise Ratio)
    Unit: dB. Measures ratio of clean signal power to residual noise power.
    Higher is better. Typically reported in the range 10–30 dB for good enhancement.

All three functions expect raw numpy float32 arrays at 16 kHz.
"""

import numpy as np
from pystoi import stoi
from pesq import pesq


def calculate_stoi(clean, enhanced, sample_rate=16000):
    """Compute STOI score between a clean reference and an enhanced signal."""
    min_len = min(len(clean), len(enhanced))
    return stoi(clean[:min_len], enhanced[:min_len], sample_rate, extended=False)


def calculate_pesq(clean, enhanced, sample_rate=16000):
    """Compute wideband PESQ score between a clean reference and an enhanced signal."""
    min_len = min(len(clean), len(enhanced))
    return pesq(sample_rate, clean[:min_len], enhanced[:min_len], "wb")


def calculate_snr(clean, enhanced):
    """Compute SNR in dB. The 'noise' is defined as (clean − enhanced)."""
    min_len = min(len(clean), len(enhanced))
    clean    = clean[:min_len]
    enhanced = enhanced[:min_len]
    noise    = clean - enhanced
    return 10.0 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))


def calculate_metrics(clean, enhanced, sample_rate=16000):
    """Compute all three metrics at once.

    Args:
        clean:       numpy array, clean reference signal
        enhanced:    numpy array, model output signal
        sample_rate: audio sample rate in Hz (default 16000)

    Returns:
        (stoi_score, pesq_score, snr_db) — all floats
    """
    stoi_score = calculate_stoi(clean, enhanced, sample_rate)
    pesq_score = calculate_pesq(clean, enhanced, sample_rate)
    snr_db     = calculate_snr(clean, enhanced)
    return stoi_score, pesq_score, snr_db
