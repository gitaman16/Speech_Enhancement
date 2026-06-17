# Interview Guide — Speech Enhancement Project

This document prepares you to explain every part of this project confidently in a placement interview, technical round, or resume walk-through. Read it once; you'll be able to answer most questions the interviewer can ask.

---

## Quick Reference: Complete Project Flow

```
Noisy .wav file
    ↓  STFT (Short-Time Fourier Transform)
Magnitude spectrogram [257 × 501]    +    Phase spectrogram (saved)
    ↓  U-Net
Enhanced magnitude spectrogram [257 × 501]
    ↓  ISTFT (combine enhanced magnitude + original phase)
Enhanced .wav file
```

The model never touches the raw waveform directly. It learns to clean spectrograms, which behave like images — which is why image segmentation architectures like U-Net work so well here.

---

## Module-by-Module Breakdown

---

### 1. `audio/stft_utils.py` — STFT Utilities

**What it does:**
Converts between the time domain (raw audio waveform) and the frequency domain (spectrogram).

**Functions:**
- `wav_to_mag_phase(waveform)` — Returns magnitude and phase spectrograms.
- `mag_phase_to_wav(mag, phase)` — Reconstructs a waveform from magnitude and phase.

**Input / Output:**
- Input: `[Batch, 64000]` float32 waveform at 16 kHz
- Output of wav_to_mag_phase: `[Batch, 257, ~501]` magnitude, `[Batch, 257, ~501]` phase
  - 257 = n_fft/2 + 1 = 512/2 + 1 (number of unique frequency bins)
  - 501 = number of time frames from a 4-second clip

**Why it exists:**
Audio signals are 1D and hard for a CNN to process directly. Converting to a 2D spectrogram lets us use the same convolutional techniques that work on images. The magnitude tells us how loud each frequency is at each moment. The phase tells us timing — we don't predict this, we just reuse it from the input.

**Key design detail:**
TensorFlow's `tf.signal.stft` returns `[Batch, time, freq]` but we transpose to `[Batch, freq, time]` to stay consistent with the original PyTorch convention. This means frequency is the height and time is the width — like a piano roll.

**Interview question:**
*"Why don't you predict the phase too?"*
Phase prediction is notoriously hard — phase wraps around (it's circular) and is perceptually less important than magnitude. Research (Griffin-Lim, PhaseNet) shows that reusing the noisy phase for reconstruction produces good results because the phase of noise is already somewhat consistent with the clean signal's timing structure.

---

### 2. `model/unet.py` — U-Net Architecture

**What it does:**
The neural network. Takes a noisy magnitude spectrogram as input and outputs a predicted clean magnitude spectrogram.

**Why U-Net?**
U-Net was invented for medical image segmentation (finding cell boundaries in microscopy images). The speech enhancement problem is structurally similar: given a "corrupted image" (noisy spectrogram), produce a "cleaned image" (clean spectrogram). Both tasks need to preserve fine spatial detail while understanding global context.

**Architecture:**
```
Encoder:   Compress the spectrogram (like reading a map from far away)
           — captures which frequencies carry noise vs speech

Bottleneck: Compact representation with the most channels
           — forces the model to learn the most important patterns

Decoder:   Expand back to original size (reconstruct the clean image)
           — each stage gets help from the matching encoder via skip connections

Skip connections: Concatenate encoder features directly into decoder
           — prevents the decoder from having to "re-learn" low-level patterns
```

**Key class: `ConvBlock`**
Two Conv2D → BatchNorm → ReLU repetitions. This is the fundamental unit of the entire network. Every encoder block, bottleneck, and decoder block is just a `ConvBlock` with different channel sizes.

**Input / Output:**
- Input:  `[Batch, 257, 501, 1]` — noisy magnitude spectrogram, single channel
- Output: `[Batch, 257, 501, 1]` — predicted clean magnitude spectrogram

**Why L1 loss (mean absolute error)?**
MSE loss (mean squared error) penalizes large errors more than small ones, which causes the model to over-smooth predictions (it "plays it safe" and predicts blurry spectrograms). L1 loss treats all errors equally and produces sharper, crisper predictions — critical for intelligibility.

**Interview question:**
*"How many parameters does the model have?"*
Approximately 7.7 million. This is mid-range for audio deep learning — large enough to learn complex noise patterns, small enough to train on a student laptop (with a GPU it trains in a few hours).

*"What do skip connections do exactly?"*
When you pool (downsample) in the encoder, you lose spatial resolution. The decoder upsamples to recover it, but some detail is permanently gone. Skip connections take the encoder feature map before pooling and paste it directly into the decoder at the matching resolution — the decoder then has access to both the abstract bottleneck features and the fine-grained early features simultaneously.

---

### 3. `data/dataset.py` — Dataset Pipeline

**What it does:**
Loads paired (noisy, clean) audio files from disk and feeds batches to the training loop.

**How it works:**
Uses `tf.data.Dataset` — TensorFlow's data pipeline framework. Files are read lazily (on demand) rather than all at once, which allows datasets larger than RAM.

Pipeline:
```
File paths list
  → .shuffle()        — randomise order each epoch
  → .map(load_pair)   — read both WAV files, fix to 64,000 samples
  → .batch(4)         — group into batches of 4
  → .prefetch()       — load next batch while GPU trains on current one
```

**Input / Output:**
- Input: two directories of `.wav` files (noisy and clean, same filenames)
- Output: batches of `(noisy, clean)` tensors, each `[4, 64000]`

**Why tf.data instead of a plain loop?**
`prefetch(AUTOTUNE)` overlaps data loading with GPU training — the GPU is never idle waiting for data. On a large dataset (5 GB+), this can cut training time by 30–50%.

**Interview question:**
*"What happens if a clip is shorter than 4 seconds?"*
We zero-pad it to exactly 64,000 samples. Zero padding adds silence at the end, which the model learns to ignore (silence has a flat, low-energy spectrogram very different from speech).

---

### 4. `train.py` — Training Loop

**What it does:**
Trains the U-Net using gradient descent, saves checkpoints.

**Training loop (one epoch):**
```
For each batch of (noisy, clean) waveforms:
  1. Compute STFT → get noisy_mag and clean_mag
  2. Add channel dim: [B, F, T] → [B, F, T, 1]
  3. Forward pass: pred_mag = model(noisy_input)
  4. Compute loss: L1(pred_mag, clean_mag)
  5. Backprop: compute gradients via GradientTape
  6. Clip gradients to norm 1.0 (prevents exploding gradients)
  7. Update weights: optimizer.apply_gradients()
```

**Optimizer:** Adam (lr = 1e-4)
Adam adapts the learning rate individually for each parameter. It's the default choice for most deep learning tasks — faster and more robust than plain SGD.

**Gradient clipping:**
Multiplies all gradients by a scaling factor if their total norm exceeds 1.0. Prevents training instability (the "exploding gradient" problem where one bad batch causes extreme weight updates).

**`@tf.function` decorator:**
Compiles the training step into a TensorFlow computation graph. First call is slow (compilation); all subsequent calls are 2–5× faster than running Python line by line.

**Interview question:**
*"Why not use model.fit()?"*
`model.fit()` is great for standard tasks. Here, each batch requires STFT preprocessing on tensors — not on raw data — before being fed to the model. A custom training loop makes this data flow explicit and easy to follow. You can read the entire training process in 40 lines.

---

### 5. `weights/convert_weights.py` — Weight Conversion

**What it does:**
Translates the trained PyTorch weights into TensorFlow format. Run once, never again.

**Why conversion is needed:**
PyTorch stores convolution kernels in `[out, in, H, W]` axis order.
TensorFlow stores them in `[H, W, in, out]` axis order.
If you load PyTorch weights directly into TensorFlow, the values are scrambled — the network produces garbage output.

**The fix:**
For every Conv2D kernel: `tf_kernel = numpy.transpose(pt_kernel, (2, 3, 1, 0))`
For every BatchNorm: direct copy (1D vectors, no reordering needed)

**The output:**
`models/unet_tf_weights.h5` — a Keras weight file that can be loaded in one line:
```python
model.load_weights("models/unet_tf_weights.h5")
```

**The original `.pt` file is never modified** — the script is read-only with respect to it.

**Interview question:**
*"Can't you just retrain in TensorFlow from scratch?"*
You could, but retraining would require the full 5 GB dataset, several hours of GPU time, and hyperparameter tuning. The existing model already achieves good PESQ/STOI scores. Conversion takes 30 seconds and preserves the trained behaviour exactly.

---

### 6. `enhance.py` — Inference Script

**What it does:**
Enhances a single noisy audio file and optionally computes quality metrics.

**Pipeline:**
```
noisy.wav → load → STFT → U-Net → ISTFT → enhanced.wav
                                 → metrics (STOI, PESQ, SNR)
                                 → plots (waveform, spectrogram, difference)
```

**Input / Output:**
- Input:  any noisy `.wav` file
- Output: enhanced `.wav` + printed metrics + 3 PNG plots

---

### 7. `evaluate.py` — Batch Evaluation

**What it does:**
Runs the model on every file in `test_set/`, prints per-file and average metrics, and saves summary charts.

**Metrics used:**
| Metric | Measures | Range | Good value |
|--------|---------|-------|------------|
| STOI   | Intelligibility | 0–1 | > 0.85 |
| PESQ   | Perceptual quality | −0.5 to 4.5 | > 2.5 |
| SNR    | Signal-to-noise ratio | dB | > 15 dB |

**Interview question:**
*"What's the difference between STOI and PESQ?"*
STOI measures whether you can understand the words (intelligibility). PESQ measures how pleasant the audio sounds to a human listener (quality). A signal can be intelligible but still sound harsh or artefact-ridden — that's why you need both.

---

### 8. `evaluation/metrics.py` — Metrics

**What it does:**
Thin wrappers around the `pystoi` and `pesq` libraries. Also implements a simple SNR calculation.

**SNR formula:**
```
SNR = 10 × log10( power(clean) / power(clean - enhanced) )
```
The "noise" is defined as the difference between clean and enhanced. If the model perfectly reconstructs the clean signal, the noise is zero and SNR → ∞.

---

### 9. `audio/slice_audio.py` — Preprocessing

**What it does:**
Slices raw long-form audio recordings into 4-second clips. Run once before training.

**Why 4 seconds?**
The U-Net input is fixed-size. 4 seconds (64,000 samples at 16 kHz) is long enough to capture natural speech patterns, short enough to fit many clips in a batch.

---

## Data Flow Through the Model (End to End)

```
noisy.wav (float32, ~64000 samples)

→ wav_to_mag_phase()
  STFT with n_fft=512, hop=128, win=512
  → magnitude [1, 257, 501]   (what the model sees)
  → phase     [1, 257, 501]   (saved, not processed)

→ tf.expand_dims(mag, axis=-1)
  → [1, 257, 501, 1]          (add channel dim for Keras)

→ UNet.call(input)
  Encoder Block 1:  [1, 257, 501, 1]   → [1, 257, 501, 16]   → pool → [1, 128, 250, 16]
  Encoder Block 2:  [1, 128, 250, 16]  → [1, 128, 250, 32]   → pool → [1, 64, 125, 32]
  Encoder Block 3:  [1, 64, 125, 32]   → [1, 64, 125, 64]    → pool → [1, 32, 62, 64]
  Encoder Block 4:  [1, 32, 62, 64]    → [1, 32, 62, 128]    → pool → [1, 16, 31, 128]
  Bottleneck:       [1, 16, 31, 128]   → [1, 16, 31, 256]
  Decoder Block 4:  upsample+concat    → [1, 32, 62, 256]    → conv → [1, 32, 62, 128]
  Decoder Block 3:  upsample+concat    → [1, 64, 125, 128]   → conv → [1, 64, 125, 64]
  Decoder Block 2:  upsample+concat    → [1, 128, 250, 64]   → conv → [1, 128, 250, 32]
  Decoder Block 1:  upsample+concat    → [1, 257, 501, 32]   → conv → [1, 257, 501, 16]
  Output conv:                                                        → [1, 257, 501, 1]

→ tf.squeeze(enhanced_mag, axis=-1)
  → [1, 257, 501]             (remove channel dim)

→ mag_phase_to_wav(enhanced_mag, original_phase)
  enhanced_stft = enhanced_mag × e^(j × phase)
  ISTFT → [1, 64000]

→ enhanced.wav
```

---

## Important Design Decisions

**1. Frequency domain, not time domain**
Working on spectrograms means each "pixel" corresponds to a specific frequency at a specific time. The CNN can learn "at 500 Hz around 1.2 seconds, there's noise" very naturally.

**2. Reusing original phase**
The phase spectrogram from the noisy audio is used unchanged. This avoids the hard problem of phase prediction and produces acceptable reconstructions because speech phase is not dramatically corrupted by additive noise.

**3. L1 loss over MSE**
L1 gives sharper spectrograms. MSE over-smooths (the model hedges its bets by predicting an average, which looks blurry).

**4. 4 skip connections**
One per encoder stage. More skip connections = more spatial detail preserved. Fewer = more compression and possible blurring.

**5. BatchNormalization in every block**
Normalises activations within each mini-batch, making training faster and more stable. Without it, deeper layers receive wildly varying input distributions and training can stall.

---

## Common Interview Questions and Suggested Answers

**Q: What is speech enhancement?**
A: Speech enhancement is the task of removing background noise from a speech signal to improve its intelligibility and perceptual quality. It has applications in hearing aids, telecommunication systems, voice assistants, and automatic speech recognition preprocessing.

**Q: Why use a U-Net for this task?**
A: U-Net combines local and global features through its encoder-decoder structure and skip connections. The encoder captures which frequency patterns correspond to noise vs speech. The skip connections ensure that fine-grained frequency detail is available in the decoder for reconstruction. This makes it particularly good at tasks that require both understanding the "big picture" (what noise sounds like globally) and preserving "local detail" (the exact shape of speech formants).

**Q: What is STFT and why is it used?**
A: STFT (Short-Time Fourier Transform) slices the audio into short overlapping windows and computes the Fourier transform of each window. This gives us a 2D time-frequency representation. CNNs are very good at processing 2D grids because of their translational equivariance — a noise pattern at one time-frequency location looks the same as at another, and the CNN learns to recognise it regardless of position.

**Q: What are skip connections and why do they matter?**
A: Skip connections copy the feature map from an encoder stage directly to the matching decoder stage. Without them, the decoder only has access to the bottleneck's compressed representation and must reconstruct fine detail from scratch — which is hard and lossy. Skip connections give the decoder a "shortcut" to the original high-resolution features.

**Q: How do you measure if the model is working?**
A: Three metrics: STOI (how intelligible the speech is, 0–1), PESQ (perceptual quality score, −0.5 to 4.5, approximating human judgements), and SNR (signal-to-noise ratio in dB). Our model improves STOI by ~0.12, PESQ by ~0.83, and SNR by ~8 dB.

**Q: What's the difference between this and a noise-cancelling headphone?**
A: Noise-cancelling headphones use adaptive filters — they measure ambient noise and produce an equal-and-opposite sound wave (active noise cancellation). This works for stationary, broadband noise (engines, hum) but fails for dynamic noise (voices, music). Our deep learning approach learns complex non-linear patterns and handles a wider variety of noise types. The downside is computational cost.

**Q: How would you deploy this model?**
A: For offline use, convert the file to TF SavedModel format and serve it via TensorFlow Serving or a FastAPI endpoint. For real-time use, implement causal (streaming) STFT — process audio frame by frame with a ring buffer. The U-Net would need modification (causal convolutions, no future context) but the principles are identical.

**Q: What would you improve?**
A: Several directions: (1) Perceptual loss function — weight frequency-domain errors by speech importance rather than treating all frequencies equally. (2) Attention mechanism in the decoder to focus on speech-dominant regions. (3) Extend to multi-channel (microphone array) input for spatial noise rejection. (4) Train on more diverse noise types (music, babble, vehicle noise) for better generalisation.

**Q: Can you explain gradient clipping?**
A: During backpropagation, gradients can occasionally be very large — this causes the optimiser to take an enormous step and destabilise training ("exploding gradients"). Gradient clipping rescales all gradients proportionally if their combined magnitude exceeds a threshold (we use 1.0). It's like a speed limiter — the model still moves in the right direction, just not too fast.

**Q: Why Adam optimiser over SGD?**
A: Adam maintains a separate adaptive learning rate for each parameter. Parameters that receive consistently large gradients (important parameters) get a smaller effective rate; rarely-updated parameters get a larger rate. This makes training significantly faster and less sensitive to the initial learning rate choice compared to SGD.

---

## 2-Minute Explanation (for HR / Non-Technical Interviewers)

"I built a deep learning system that removes background noise from speech recordings — similar to what noise-cancelling headphones do, but using a neural network instead of hardware.

The key idea is that instead of processing raw audio, we first convert it into a 2D image called a spectrogram — a visual representation of which frequencies are present at which times, like a musical score. We then train a convolutional neural network — the same type of model used for image recognition — to clean up this spectrogram by removing the noise patterns it has learned to recognise.

The model, called a U-Net, was originally designed for medical image analysis. I adapted it here and trained it on thousands of examples of noisy and clean speech pairs. After training, given any noisy recording, it predicts what the clean version would look like.

The system measurably improves speech quality — we tested it with standard metrics used in the telecommunications industry and showed consistent improvements in both intelligibility and perceptual quality."

---

## 5-Minute Explanation (for Technical Interviewers)

"The project is a supervised speech enhancement system using a U-Net trained on paired noisy/clean speech.

**Data preparation:** I used the DNS Challenge dataset — about 5 GB of clean speech mixed with various noise types at random SNRs. Long recordings are sliced into 4-second clips. The dataset has roughly 100,000 pairs.

**Preprocessing:** Each waveform is converted to a magnitude spectrogram using the STFT with n_fft=512, hop=128. This gives a [257 × 501] image representing 257 frequency bins over ~500 time frames. The phase is also extracted and stored — we only process the magnitude, not the phase.

**Model:** The U-Net has 4 encoder blocks, a bottleneck, and 4 decoder blocks. Each block contains two Conv2D → BatchNorm → ReLU operations. The encoder progressively halves spatial resolution while doubling channels — 1→16→32→64→128→256. The decoder mirrors this, upsampling with transposed convolutions and concatenating skip connections from the encoder. The final 1×1 conv produces a single-channel output.

**Loss and training:** We use L1 loss between the predicted and target magnitude spectrograms. L1 is preferred over MSE because it produces sharper predictions — MSE tends to over-smooth. Adam optimiser at 1e-4, batch size 4, 45 total epochs with gradient clipping at norm 1.0.

**Inference:** At inference, the noisy magnitude is fed through the U-Net, the enhanced magnitude is combined with the original noisy phase via complex multiplication, and ISTFT reconstructs the enhanced waveform.

**Results:** On 30 held-out test files, we achieved a STOI improvement of 0.12, PESQ improvement of 0.83, and SNR improvement of ~8 dB.

**Codebase:** Written in TensorFlow/Keras. The original model was in PyTorch — I wrote a weight conversion script that handles the axis-order difference between the two frameworks, so the trained weights are preserved without retraining."

---

## Resume Description

```
Speech Enhancement System | Python, TensorFlow, Keras, Signal Processing | 2024

• Designed and trained a U-Net neural network (~7.7M parameters) to remove
  background noise from speech recordings using spectrogram-domain processing.

• Implemented a complete audio ML pipeline: STFT preprocessing, custom
  tf.data training pipeline with prefetching, gradient-clipped training loop,
  checkpoint management, and ISTFT reconstruction.

• Achieved measurable improvements over baseline noisy audio:
  STOI +0.12, PESQ +0.83, SNR +8.4 dB on a 30-sample held-out test set.

• Authored a PyTorch-to-TensorFlow weight conversion script handling
  kernel axis-order differences (NCHW → NHWC), preserving trained weights
  without retraining.

• Dataset: Microsoft DNS Challenge (~100K pairs, 5 GB). Preprocessing
  includes automated slicing of long recordings into fixed-length segments.
```
