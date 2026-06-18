"""
STFT utility functions for converting between waveforms and spectrograms.

The U-Net processes audio in the frequency domain. These two functions form
the bridge between the waveform (what we hear) and the spectrogram (what the
model sees).

Key difference from PyTorch:
    tf.signal.stft  returns [B, time_frames, freq_bins]  (time-major)
    torch.stft      returns [B, freq_bins, time_frames]  (freq-major)

We transpose after STFT so the rest of the code uses [B, freq_bins, time_frames]
consistently, matching the original PyTorch convention.
"""

import tensorflow as tf

# Default STFT parameters — must match the values used during training
N_FFT      = 512   # FFT size → freq_bins = N_FFT // 2 + 1 = 257
HOP_LENGTH = 128   # hop between frames (controls time resolution)
WIN_LENGTH = 512   # analysis window length


def wav_to_mag_phase(waveform, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH):
    """Convert a batch of waveforms into magnitude and phase spectrograms.

    Args:
        waveform: float32 tensor of shape [B, T] or [T]
        n_fft:    FFT size (controls frequency resolution)
        hop:      hop length between successive STFT frames
        win:      analysis window length

    Returns:
        mag:   float32 tensor [B, freq_bins, time_frames] — amplitude at each freq/time
        phase: float32 tensor [B, freq_bins, time_frames] — phase angle, kept for reconstruction
    """
    if len(waveform.shape) == 1:
        waveform = tf.expand_dims(waveform, axis=0)

    # tf.signal.stft output shape: [B, time_frames, freq_bins]
    stft_output = tf.signal.stft(
        waveform,
        frame_length=win,
        frame_step=hop,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window
    )

    # Reorder to [B, freq_bins, time_frames] to match the U-Net's spatial convention
    stft_output = tf.transpose(stft_output, perm=[0, 2, 1])

    mag   = tf.abs(stft_output)
    phase = tf.math.angle(stft_output)

    return mag, phase


def mag_phase_to_wav(mag, phase, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH, target_len=None):
    """Reconstruct a waveform from magnitude and phase spectrograms.

    The U-Net predicts an enhanced magnitude. We combine it with the original
    noisy phase (which carries timing/structural information) to rebuild the
    complex STFT, then invert it back to a waveform.

    Args:
        mag:        float32 tensor [B, freq_bins, time_frames]
        phase:      float32 tensor [B, freq_bins, time_frames] (from the original noisy audio)
        n_fft:      FFT size (must match what was used in wav_to_mag_phase)
        hop:        hop length
        win:        window length
        target_len: optional int — trim output to exactly this many samples

    Returns:
        wav: float32 tensor [B, T]
    """
    # Combine enhanced magnitude with original phase: complex = mag * e^(j*phase)
    min_freq = tf.minimum(tf.shape(mag)[1], tf.shape(phase)[1])
    min_time = tf.minimum(tf.shape(mag)[2], tf.shape(phase)[2])
    mag   = mag[:, :min_freq, :min_time]
    phase = phase[:, :min_freq, :min_time]
    
    real = mag * tf.cos(phase)
    imag = mag * tf.sin(phase)
    stft_complex = tf.complex(real, imag)  # [B, freq_bins, time_frames]

    # Reorder back to [B, time_frames, freq_bins] as expected by tf.signal.inverse_stft
    stft_complex = tf.transpose(stft_complex, perm=[0, 2, 1])

    inverse_window_fn = tf.signal.inverse_stft_window_fn(
        frame_step=hop,
        forward_window_fn=tf.signal.hann_window
    )

    wav = tf.signal.inverse_stft(
        stfts=stft_complex,
        frame_length=win,
        frame_step=hop,
        fft_length=n_fft,
        window_fn=inverse_window_fn
    )

    if target_len is not None:
        wav = wav[:, :target_len]

    return wav
