"""
Dataset pipeline for speech enhancement training.

Replaces PyTorch's Dataset + DataLoader with a tf.data pipeline.
The pipeline reads paired (noisy, clean) audio files, pads or trims each
clip to a fixed length, and batches them for training.

Directory convention (both directories must have the same number of files,
sorted so that noisy[i] corresponds to clean[i]):

    data/
    ├── noisy_4s/   ← noisy speech clips (e.g. speech + background noise)
    └── clean_4s/   ← matching clean speech clips
"""

import os
import glob

import numpy as np
import soundfile as sf
import tensorflow as tf

MAX_AUDIO_LEN = 64000   # 4 seconds at 16 kHz


def _fix_length(audio, target_len):
    """Pad with zeros or trim to reach exactly target_len samples."""
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def _load_pair(noisy_path, clean_path, max_len):
    """Load one (noisy, clean) audio pair and fix both to max_len samples."""
    noisy, _ = sf.read(noisy_path.decode())
    clean, _ = sf.read(clean_path.decode())
    noisy = _fix_length(noisy, max_len)
    clean = _fix_length(clean, max_len)
    return noisy, clean


def make_dataset(noisy_dir, clean_dir, batch_size=4, max_len=MAX_AUDIO_LEN, shuffle=True):
    """Build a tf.data.Dataset for training or evaluation.

    Args:
        noisy_dir:  path to directory of noisy .wav files
        clean_dir:  path to directory of clean .wav files
        batch_size: number of audio clips per batch
        max_len:    clip length in samples (pad/trim to this)
        shuffle:    whether to shuffle the dataset each epoch

    Returns:
        tf.data.Dataset yielding (noisy_batch, clean_batch) tuples,
        each of shape [batch_size, max_len]
    """
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))

    if len(noisy_files) == 0:
        raise FileNotFoundError(f"No .wav files found in: {noisy_dir}")
    if len(noisy_files) != len(clean_files):
        raise ValueError(
            f"Noisy ({len(noisy_files)}) and clean ({len(clean_files)}) "
            "directories must have the same number of files."
        )

    print(f"  Found {len(noisy_files)} audio pairs in dataset.")

    noisy_paths = tf.constant(noisy_files)
    clean_paths = tf.constant(clean_files)

    path_dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))

    if shuffle:
        path_dataset = path_dataset.shuffle(buffer_size=len(noisy_files), reshuffle_each_iteration=True)

    def load_pair_tf(noisy_path, clean_path):
        noisy, clean = tf.numpy_function(
            func=lambda n, c: _load_pair(n, c, max_len),
            inp=[noisy_path, clean_path],
            Tout=[tf.float32, tf.float32]
        )
        noisy.set_shape([max_len])
        clean.set_shape([max_len])
        return noisy, clean

    dataset = (
        path_dataset
        .map(load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset
