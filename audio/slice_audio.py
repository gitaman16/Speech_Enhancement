"""
Preprocessing script: slice long audio files into fixed-length segments.

The U-Net expects exactly 4-second clips (64,000 samples at 16 kHz).
Run this script once on your raw dataset before training.

Usage:
    python audio/slice_audio.py

The script reads paired (clean, noisy) audio files, slices them into
4-second segments, zero-pads the final short segment, and saves the
results to output directories.

Input structure:
    data/raw/clean/    ← original clean recordings (any length)
    data/raw/noisy/    ← matching noisy recordings

Output structure:
    data/clean_4s/     ← clean 4-second clips
    data/noisy_4s/     ← noisy 4-second clips
"""

import os
import numpy as np
import soundfile as sf


def slice_audio(
    input_clean_dir,
    input_noisy_dir,
    output_clean_dir,
    output_noisy_dir,
    segment_seconds=4,
    sample_rate=16000
):
    """Slice all paired audio files into fixed-length segments.

    Args:
        input_clean_dir:   directory containing clean recordings
        input_noisy_dir:   directory containing matching noisy recordings
        output_clean_dir:  where to save clean segments
        output_noisy_dir:  where to save noisy segments
        segment_seconds:   length of each output clip in seconds
        sample_rate:       audio sample rate (must match recording)
    """
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    clean_files = sorted([f for f in os.listdir(input_clean_dir) if f.endswith(".wav")])
    noisy_files = sorted([f for f in os.listdir(input_noisy_dir) if f.endswith(".wav")])

    if len(clean_files) != len(noisy_files):
        raise ValueError(
            f"Mismatch: {len(clean_files)} clean files vs {len(noisy_files)} noisy files."
        )

    segment_len = segment_seconds * sample_rate
    total_segments = 0

    for clean_filename, noisy_filename in zip(clean_files, noisy_files):
        clean_audio, _ = sf.read(os.path.join(input_clean_dir, clean_filename))
        noisy_audio, _ = sf.read(os.path.join(input_noisy_dir, noisy_filename))

        stem = os.path.splitext(clean_filename)[0]

        for start in range(0, len(clean_audio), segment_len):
            clean_segment = clean_audio[start : start + segment_len]
            noisy_segment = noisy_audio[start : start + segment_len]

            # Zero-pad the final segment if it's shorter than segment_len
            if len(clean_segment) < segment_len:
                clean_segment = np.pad(clean_segment, (0, segment_len - len(clean_segment)))
                noisy_segment = np.pad(noisy_segment, (0, segment_len - len(noisy_segment)))

            sf.write(os.path.join(output_clean_dir, f"{stem}_{start}.wav"), clean_segment, sample_rate)
            sf.write(os.path.join(output_noisy_dir, f"{stem}_{start}.wav"), noisy_segment, sample_rate)
            total_segments += 1

    print(f"Done. {total_segments} segments saved.")


if __name__ == "__main__":
    slice_audio(
        input_clean_dir="data/raw/clean",
        input_noisy_dir="data/raw/noisy",
        output_clean_dir="data/clean_4s",
        output_noisy_dir="data/noisy_4s",
    )
