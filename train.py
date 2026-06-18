"""
Training script for the U-Net Speech Enhancement model.

Workflow:
    1. Load paired (noisy, clean) audio clips
    2. Build the U-Net model
    3. For each batch: compute STFT → run model → compute loss → backprop
    4. Save a checkpoint every CHECKPOINT_INTERVAL epochs
    5. Save final weights when done

Resume training:
    Set RESUME_FROM to a checkpoint .h5 path to continue from that point.
    The model will train for NUM_EPOCHS additional epochs on top of that checkpoint.

Usage:
    python train.py
"""

import os
import sys
import tensorflow as tf
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.unet       import build_unet
from audio.stft_utils import wav_to_mag_phase
from data.dataset     import make_dataset


# ── Hyperparameters ───────────────────────────────────────────────────────────
NOISY_DIR            = "data/noisy_4s"
CLEAN_DIR            = "data/clean_4s"
BATCH_SIZE           = 4
LEARNING_RATE        = 1e-4
NUM_EPOCHS           = 30
CHECKPOINT_INTERVAL  = 5        # save weights every N epochs
GRADIENT_CLIP_NORM   = 1.0
WEIGHTS_DIR          = "models"

# Set to a .h5 path to resume training from a checkpoint, otherwise None
RESUME_FROM          = None     # e.g. "models/checkpoint_epoch_15.h5"


# ── Training step ─────────────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, noisy_batch, clean_batch):
    """Run one forward + backward pass and return the batch loss.

    This function is decorated with @tf.function so TensorFlow compiles it
    into a computation graph for maximum speed on GPU/TPU.

    Steps:
        1. Convert raw waveforms to magnitude spectrograms via STFT
        2. Feed noisy magnitude through the model
        3. Compute L1 loss between predicted and clean magnitudes
        4. Compute gradients and update model weights
    """
    with tf.GradientTape() as tape:
        # Convert waveforms → spectrograms
        noisy_mag, _ = wav_to_mag_phase(noisy_batch)
        clean_mag, _ = wav_to_mag_phase(clean_batch)

        # Add channel dimension: [B, F, T] → [B, F, T, 1]
        noisy_input  = tf.expand_dims(noisy_mag, axis=-1)
        clean_target = tf.expand_dims(clean_mag, axis=-1)

        # Forward pass
        pred_mag = model(noisy_input, training=True)

        # Trim both tensors to the same size. Tiny frame count differences can
        # arise because STFT frame count depends on exact signal length.
        min_freq = tf.minimum(tf.shape(pred_mag)[1], tf.shape(clean_target)[1])
        min_time = tf.minimum(tf.shape(pred_mag)[2], tf.shape(clean_target)[2])
        pred_mag     = pred_mag[:, :min_freq, :min_time, :]
        clean_target = clean_target[:, :min_freq, :min_time, :]

        # L1 loss: encourages sharp spectrogram prediction with fewer artefacts
        # than MSE, which tends to over-smooth high-frequency content.
        loss = tf.reduce_mean(tf.abs(pred_mag - clean_target))

    gradients = tape.gradient(loss, model.trainable_variables)

    # Gradient clipping prevents exploding gradients in early training
    gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_NORM)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── GPU info ──────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    device_name = f"{len(gpus)} GPU(s)" if gpus else "CPU"
    print("=" * 60)
    print(f"  Device:        {device_name}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs:        {NUM_EPOCHS}")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_dataset = make_dataset(
        noisy_dir=NOISY_DIR,
        clean_dir=CLEAN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ── Model and optimizer ───────────────────────────────────────────────────
    model     = build_unet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Initialize weights with a dummy forward pass (TF lazy-initializes layers)
    dummy = tf.zeros([1, 257, 501, 1], dtype=tf.float32)
    model(dummy, training=False)
    print(f"  Model parameters: {model.count_params():,}")

    # ── Optional: resume from checkpoint ─────────────────────────────────────
    start_epoch = 0
    if RESUME_FROM:
        if not os.path.exists(RESUME_FROM):
            sys.exit(f"Checkpoint not found: {RESUME_FROM}")
        model.load_weights(RESUME_FROM)
        # Parse epoch number from filename, e.g. "models/checkpoint_epoch_15.h5"
        try:
            start_epoch = int(RESUME_FROM.split("epoch_")[-1].replace(".h5", ""))
        except ValueError:
            start_epoch = 0
        print(f"  Resumed from: {RESUME_FROM}  (starting at epoch {start_epoch + 1})")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        epoch_loss   = 0.0
        num_batches  = 0

        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1:3d}/{start_epoch + NUM_EPOCHS}")

        for noisy_batch, clean_batch in progress_bar:
            batch_loss = train_step(model, optimizer, noisy_batch, clean_batch)
            epoch_loss  += batch_loss.numpy()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{batch_loss.numpy():.4f}"})

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1:3d} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(WEIGHTS_DIR, f"checkpoint_epoch_{epoch + 1}.weights.h5")
            model.save_weights(ckpt_path)
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

    # ── Save final weights ────────────────────────────────────────────────────
    final_path = os.path.join(WEIGHTS_DIR, "unet_tf_weights.weights.h5")
    model.save_weights(final_path)

    print("\n" + "=" * 60)
    print(f"  Training complete.")
    print(f"  Final weights → {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
