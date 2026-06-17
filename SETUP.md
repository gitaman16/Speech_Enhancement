# Setup & Run Guide

Complete instructions to go from zero to running inference, evaluation, and training.

---

## Prerequisites

- Python 3.9 or later
- pip
- (Recommended) NVIDIA GPU with CUDA for training; CPU works fine for inference

---

## Step 1 — Set Up Environment

```bash
# Create a clean virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 2 — Copy Your Original PyTorch Weights

Copy `unet_final_continued.pt` from your original project into the `models/` folder:

```
speech_enhancement/
└── models/
    └── unet_final_continued.pt   ← paste it here
```

---

## Step 3 — Convert Weights (One-Time Setup)

This converts the PyTorch checkpoint into a TensorFlow `.h5` file.

```bash
# Install PyTorch (CPU-only is enough — no GPU needed for conversion)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run the conversion
python weights/convert_weights.py
```

Expected output:
```
============================================================
PyTorch → TensorFlow Weight Conversion
============================================================
  Loaded PyTorch checkpoint (epoch 45)
  TF model built: 7,759,105 parameters

  Copying encoder blocks...
    encoder_block1 (1→16)   ✓
    ...
  Copying bottleneck...
    bottleneck (128→256) ✓
  ...
  Running sanity check...
    Input:  [1, 257, 501, 1]
    Output: [1, 257, 501, 1]  ✓

  Saved TF weights → models/unet_tf_weights.h5
============================================================
```

After this, PyTorch is no longer needed.

---

## Step 4A — Enhance a Single File (Inference)

```bash
# Basic: enhance only
python enhance.py \
  --input  test_audio/noisy.wav \
  --output test_audio/enhanced.wav

# Full: enhance + quality metrics + plots
python enhance.py \
  --input  test_audio/noisy.wav \
  --output test_audio/enhanced.wav \
  --clean  test_audio/clean.wav
```

Output:
- `test_audio/enhanced.wav` — the cleaned audio file
- `waveform_comparison.png` — noisy vs enhanced waveform
- `spectrogram_comparison.png` — noisy vs enhanced spectrogram
- `difference_spectrogram.png` — what the model removed
- Printed STOI / PESQ / SNR scores (if `--clean` is provided)

---

## Step 4B — Batch Evaluation on Test Set

Put your test files into:
```
test_set/
├── clean/   ← ground truth clean .wav files
└── noisy/   ← matching noisy .wav files (same filenames)
```

Then run:

```bash
python evaluate.py
```

Output:
- Per-file metrics table printed to terminal
- Average STOI / PESQ / SNR across all files
- `avg_metrics_comparison.png`
- `improvement_stoi.png`, `improvement_pesq.png`, `improvement_snr.png`
- `histogram_stoi.png`, `histogram_pesq.png`, `histogram_snr.png`

---

## Step 5 — Preprocess Data (Only if Training from Scratch)

If you have raw long-form audio recordings, slice them into 4-second clips:

```bash
# Edit the paths at the bottom of audio/slice_audio.py first, then:
python audio/slice_audio.py
```

Expects:
```
data/raw/clean/   ← clean recordings (.wav)
data/raw/noisy/   ← matching noisy recordings (.wav)
```

Outputs:
```
data/clean_4s/   ← 4-second clean clips
data/noisy_4s/   ← 4-second noisy clips
```

---

## Step 6 — Train from Scratch (Optional)

Edit the paths and hyperparameters at the top of `train.py`:

```python
NOISY_DIR   = "data/noisy_4s"   # your noisy clips
CLEAN_DIR   = "data/clean_4s"   # your clean clips
BATCH_SIZE  = 4
NUM_EPOCHS  = 30
RESUME_FROM = None              # set to a .h5 path to resume
```

Then:

```bash
python train.py
```

Checkpoints are saved every 5 epochs to `models/checkpoint_epoch_N.h5`.
Final weights saved to `models/unet_tf_weights.h5`.

To resume from epoch 15:
```python
RESUME_FROM = "models/checkpoint_epoch_15.h5"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` |
| `Weights not found: models/unet_tf_weights.h5` | Run `python weights/convert_weights.py` first |
| `No .wav files found in: data/noisy_4s` | Run `python audio/slice_audio.py` to prepare data |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `train.py` to 2 |
| PESQ / STOI import errors | Run `pip install pystoi pesq` |
| Plots don't display (headless server) | Add `matplotlib.use('Agg')` at top of script |

---

## File Roles at a Glance

| File | When to run |
|------|------------|
| `audio/slice_audio.py` | Once, before training, to prepare data |
| `weights/convert_weights.py` | Once, to convert PyTorch → TF weights |
| `train.py` | To train or fine-tune the model |
| `enhance.py` | To enhance a single noisy file |
| `evaluate.py` | To measure model performance on a test set |
