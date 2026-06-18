"""
Batch evaluation script: measure model performance across a full test set.

Compares noisy audio vs. model-enhanced audio against clean references.
Produces per-file metrics, aggregate averages, and summary charts.

Expected directory structure:
    test_set/
    ├── clean/    ← clean reference .wav files (ground truth)
    └── noisy/    ← noisy input .wav files (same filenames as clean/)

Usage:
    python evaluate.py
"""

import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.unet         import build_unet
from audio.stft_utils   import wav_to_mag_phase, mag_phase_to_wav
from evaluation.metrics import calculate_metrics

WEIGHTS_PATH   = "models/unet_tf_weights.weights.h5"
TEST_CLEAN_DIR = "test_set/clean"
TEST_NOISY_DIR = "test_set/noisy"
SAMPLE_RATE    = 16000


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path):
    model = build_unet()
    dummy = tf.zeros([1, 257, 501, 1], dtype=tf.float32)
    model(dummy, training=False)
    model.load_weights(weights_path)
    return model


# ── Single-file enhancement ───────────────────────────────────────────────────

def enhance_waveform(model, noisy_wav):
    """Enhance a single numpy waveform. Returns enhanced numpy array."""
    original_len  = len(noisy_wav)
    noisy_tensor  = tf.constant(noisy_wav[np.newaxis, :], dtype=tf.float32)
    mag, phase    = wav_to_mag_phase(noisy_tensor)
    mag_input     = tf.expand_dims(mag, axis=-1)
    enhanced_mag  = tf.squeeze(model(mag_input, training=False), axis=-1)
    enhanced_wav  = mag_phase_to_wav(enhanced_mag, phase, target_len=original_len)
    return enhanced_wav[0].numpy()


# ── Plotting utilities ────────────────────────────────────────────────────────

def plot_average_metrics(noisy_scores, enhanced_scores, metric_names):
    """Bar chart comparing average noisy vs enhanced scores for each metric."""
    num_metrics = len(metric_names)
    x = np.arange(num_metrics)
    width = 0.35

    noisy_means    = [np.mean(noisy_scores[m])    for m in metric_names]
    enhanced_means = [np.mean(enhanced_scores[m]) for m in metric_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, noisy_means,    width, label="Noisy",    color="steelblue",  alpha=0.85)
    ax.bar(x + width / 2, enhanced_means, width, label="Enhanced", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Average Quality Metrics: Noisy vs Enhanced", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("avg_metrics_comparison.png", dpi=150)
    plt.show()
    print("  Saved: avg_metrics_comparison.png")


def plot_improvement_per_file(noisy_scores, enhanced_scores, metric_name):
    """Line plot showing per-file improvement (enhanced − noisy)."""
    improvements = np.array(enhanced_scores[metric_name]) - np.array(noisy_scores[metric_name])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(improvements, marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"{metric_name} Improvement Per File (Enhanced − Noisy)", fontsize=12)
    ax.set_xlabel("File index")
    ax.set_ylabel(f"Δ {metric_name}")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"improvement_{metric_name.lower()}.png", dpi=150)
    plt.show()
    print(f"  Saved: improvement_{metric_name.lower()}.png")


def plot_improvement_histogram(noisy_scores, enhanced_scores, metric_name):
    """Histogram of per-file improvements."""
    improvements = np.array(enhanced_scores[metric_name]) - np.array(noisy_scores[metric_name])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(improvements, bins=12, alpha=0.85, color="teal", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"{metric_name} Improvement Distribution", fontsize=12)
    ax.set_xlabel(f"Δ {metric_name}")
    ax.set_ylabel("Number of files")
    plt.tight_layout()
    plt.savefig(f"histogram_{metric_name.lower()}.png", dpi=150)
    plt.show()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate():
    if not os.path.exists(WEIGHTS_PATH):
        sys.exit(
            f"Weights not found: {WEIGHTS_PATH}\n"
            "Run weights/convert_weights.py first, or train with train.py."
        )

    print("=" * 60)
    print("  Batch Evaluation")
    print(f"  Test clean: {TEST_CLEAN_DIR}")
    print(f"  Test noisy: {TEST_NOISY_DIR}")
    print("=" * 60 + "\n")

    model = load_model(WEIGHTS_PATH)

    clean_files = sorted(f for f in os.listdir(TEST_CLEAN_DIR) if f.endswith(".wav"))
    noisy_files = sorted(f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav"))

    if not clean_files:
        sys.exit(f"No .wav files found in {TEST_CLEAN_DIR}")

    # Match files by name — both directories should have the same filenames
    matched = [(f, f) for f in clean_files if f in noisy_files]
    if not matched:
        # Fallback: match by order
        matched = list(zip(clean_files, noisy_files))
        print("  Warning: filenames differ between clean/noisy dirs — matching by sort order.\n")

    print(f"  Evaluating {len(matched)} file pairs...\n")

    # Accumulate scores across all files
    noisy_scores    = {"STOI": [], "PESQ": [], "SNR": []}
    enhanced_scores = {"STOI": [], "PESQ": [], "SNR": []}

    for i, (clean_fname, noisy_fname) in enumerate(matched):
        clean, _ = sf.read(os.path.join(TEST_CLEAN_DIR, clean_fname))
        noisy, _ = sf.read(os.path.join(TEST_NOISY_DIR, noisy_fname))

        if clean.ndim > 1: clean = clean[:, 0]
        if noisy.ndim > 1: noisy = noisy[:, 0]
        clean = clean.astype(np.float32)
        noisy = noisy.astype(np.float32)

        enhanced = enhance_waveform(model, noisy)

        min_len  = min(len(clean), len(noisy), len(enhanced))
        clean    = clean[:min_len]
        noisy    = noisy[:min_len]
        enhanced = enhanced[:min_len]

        # Metrics for noisy vs clean (baseline — how bad was it before?)
        stoi_n, pesq_n, snr_n = calculate_metrics(clean, noisy, SAMPLE_RATE)
        # Metrics for enhanced vs clean (result — how good is it after?)
        stoi_e, pesq_e, snr_e = calculate_metrics(clean, enhanced, SAMPLE_RATE)

        noisy_scores["STOI"].append(stoi_n)
        noisy_scores["PESQ"].append(pesq_n)
        noisy_scores["SNR"].append(snr_n)

        enhanced_scores["STOI"].append(stoi_e)
        enhanced_scores["PESQ"].append(pesq_e)
        enhanced_scores["SNR"].append(snr_e)

        print(f"  [{i + 1:3d}] {clean_fname}")
        print(f"         Noisy   →  STOI: {stoi_n:.4f}  PESQ: {pesq_n:.4f}  SNR: {snr_n:.2f} dB")
        print(f"         Enhanced → STOI: {stoi_e:.4f}  PESQ: {pesq_e:.4f}  SNR: {snr_e:.2f} dB\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  AVERAGE RESULTS")
    print("=" * 60)

    for metric in ["STOI", "PESQ", "SNR"]:
        n_mean = np.mean(noisy_scores[metric])
        e_mean = np.mean(enhanced_scores[metric])
        unit   = " dB" if metric == "SNR" else ""
        print(f"  {metric:4s}  Noisy: {n_mean:.4f}{unit}  →  Enhanced: {e_mean:.4f}{unit}"
              f"  (Δ {e_mean - n_mean:+.4f}{unit})")

    print("=" * 60 + "\n")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("  Generating charts...")
    plot_average_metrics(noisy_scores, enhanced_scores, ["STOI", "PESQ", "SNR"])
    for metric in ["STOI", "PESQ", "SNR"]:
        plot_improvement_per_file(noisy_scores, enhanced_scores, metric)
        plot_improvement_histogram(noisy_scores, enhanced_scores, metric)

    print("\n  Evaluation complete.")


if __name__ == "__main__":
    evaluate()
