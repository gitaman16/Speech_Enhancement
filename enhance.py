"""
Inference script: enhance a single noisy audio file.

Loads the trained U-Net weights, runs the full enhancement pipeline, saves
the enhanced audio, prints quality metrics (if a clean reference is given),
and generates spectrogram comparison plots.

Usage:
    # Enhance only (no metrics)
    python enhance.py --input test_audio/noisy.wav --output test_audio/enhanced.wav

    # Enhance + compute metrics against clean reference
    python enhance.py --input test_audio/noisy.wav \
                      --output test_audio/enhanced.wav \
                      --clean  test_audio/clean.wav
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.unet          import build_unet
from audio.stft_utils    import wav_to_mag_phase, mag_phase_to_wav
from evaluation.metrics  import calculate_metrics

WEIGHTS_PATH = "models/unet_tf_weights.weights.h5"
SAMPLE_RATE  = 16000


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path):
    """Build the U-Net and restore trained weights."""
    model = build_unet()

    # Dummy forward pass to trigger lazy weight initialization
    dummy = tf.zeros([1, 257, 501, 1], dtype=tf.float32)
    model(dummy, training=False)

    model.load_weights(weights_path)
    print(f"  Model loaded from: {weights_path}")
    return model


# ── Enhancement pipeline ──────────────────────────────────────────────────────

def enhance(model, noisy_wav):
    """Run the full enhancement pipeline on a single waveform.

    Pipeline:
        waveform → STFT → magnitude + phase → U-Net → enhanced magnitude
        → ISTFT (with original phase) → enhanced waveform

    Args:
        model:     trained Keras U-Net
        noisy_wav: numpy float32 array of shape [T]

    Returns:
        enhanced numpy float32 array of shape [T]
    """
    original_len = len(noisy_wav)

    # Add batch dimension: [T] → [1, T]
    noisy_tensor = tf.constant(noisy_wav[np.newaxis, :], dtype=tf.float32)

    # Waveform → magnitude + phase spectrograms
    mag, phase = wav_to_mag_phase(noisy_tensor)

    # Add channel dimension for U-Net: [1, F, T] → [1, F, T, 1]
    mag_input = tf.expand_dims(mag, axis=-1)

    # U-Net predicts the enhanced magnitude
    enhanced_mag = model(mag_input, training=False)

    # Remove channel dimension: [1, F, T, 1] → [1, F, T]
    enhanced_mag = tf.squeeze(enhanced_mag, axis=-1)

    # Reconstruct waveform from enhanced magnitude + original phase
    enhanced_wav = mag_phase_to_wav(enhanced_mag, phase, target_len=original_len)

    return enhanced_wav[0].numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spectrogram(signal, sr, title, ax):
    """Draw a log-magnitude spectrogram onto the given matplotlib axis."""
    _, _, Zxx = scipy_stft(signal, sr, nperseg=512, noverlap=256)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-8)
    img = ax.imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest"
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Frequency bins")
    plt.colorbar(img, ax=ax, label="dB")


def plot_results(noisy, enhanced, clean=None, sr=SAMPLE_RATE):
    """Generate waveform and spectrogram comparison plots."""
    has_clean = clean is not None
    num_specs = 3 if has_clean else 2

    # ── Waveform comparison ───────────────────────────────────────────────────
    fig_wave, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.arange(len(noisy)) / sr
    ax.plot(time_axis, noisy,    label="Noisy",    alpha=0.7, linewidth=0.8)
    ax.plot(time_axis, enhanced, label="Enhanced", alpha=0.7, linewidth=0.8)
    if has_clean:
        ax.plot(time_axis, clean, label="Clean", alpha=0.7, linewidth=0.8)
    ax.set_title("Waveform Comparison", fontsize=14)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig("waveform_comparison.png", dpi=150)
    plt.show()

    # ── Spectrogram comparison ────────────────────────────────────────────────
    fig_spec, axes = plt.subplots(1, num_specs, figsize=(5 * num_specs, 4))
    plot_spectrogram(noisy,    sr, "Noisy",    axes[0])
    plot_spectrogram(enhanced, sr, "Enhanced", axes[1])
    if has_clean:
        plot_spectrogram(clean, sr, "Clean", axes[2])
    plt.suptitle("Spectrogram Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("spectrogram_comparison.png", dpi=150)
    plt.show()

    # ── Difference spectrogram (shows what the model removed) ─────────────────
    fig_diff, ax = plt.subplots(figsize=(10, 4))
    _, _, S_noisy = scipy_stft(noisy,    sr, nperseg=512, noverlap=256)
    _, _, S_enh   = scipy_stft(enhanced, sr, nperseg=512, noverlap=256)
    min_t = min(S_noisy.shape[1], S_enh.shape[1])
    diff_db = (
        20 * np.log10(np.abs(S_enh[:, :min_t]) + 1e-8)
        - 20 * np.log10(np.abs(S_noisy[:, :min_t]) + 1e-8)
    )
    img = ax.imshow(diff_db, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(img, ax=ax, label="dB difference")
    ax.set_title("Difference Spectrogram: Enhanced − Noisy\n(blue = noise removed, red = boosted)", fontsize=12)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig("difference_spectrogram.png", dpi=150)
    plt.show()

    print("  Plots saved: waveform_comparison.png, spectrogram_comparison.png, difference_spectrogram.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Enhance a noisy audio file.")
    parser.add_argument("--input",   required=True, help="Path to noisy input .wav file")
    parser.add_argument("--output",  required=True, help="Path to save enhanced .wav file")
    parser.add_argument("--clean",   default=None,  help="(Optional) Clean reference .wav for metrics")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Path to .h5 weights file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Speech Enhancement Inference")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.weights):
        sys.exit(
            f"Weights not found: {args.weights}\n"
            "Run weights/convert_weights.py first, or train from scratch with train.py."
        )
    model = load_model(args.weights)

    # ── Load audio ────────────────────────────────────────────────────────────
    noisy, sr = sf.read(args.input)
    if noisy.ndim > 1:
        noisy = noisy[:, 0]  # take first channel if stereo
    noisy = noisy.astype(np.float32)
    print(f"  Input:  {args.input}  ({len(noisy)/sr:.1f}s, {sr}Hz)")

    # ── Enhance ───────────────────────────────────────────────────────────────
    print("  Enhancing...")
    enhanced = enhance(model, noisy)

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, enhanced, sr)
    print(f"  Output: {args.output}")

    # ── Metrics (optional) ────────────────────────────────────────────────────
    clean = None
    if args.clean:
        clean, _ = sf.read(args.clean)
        if clean.ndim > 1:
            clean = clean[:, 0]
        clean = clean.astype(np.float32)

        min_len = min(len(clean), len(enhanced))
        stoi_score, pesq_score, snr_score = calculate_metrics(
            clean[:min_len], enhanced[:min_len], sr
        )
        print("\n  ─── Quality Metrics ───────────────────────────────────")
        print(f"  STOI  : {stoi_score:.4f}   (0–1, higher = more intelligible)")
        print(f"  PESQ  : {pesq_score:.4f}   (−0.5 to 4.5, higher = better quality)")
        print(f"  SNR   : {snr_score:.2f} dB  (higher = less residual noise)")
        print("  ───────────────────────────────────────────────────────")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    min_len = min(len(noisy), len(enhanced))
    plot_results(noisy[:min_len], enhanced[:min_len], clean[:min_len] if clean is not None else None, sr)

    print("\n  Done.")


if __name__ == "__main__":
    main()
