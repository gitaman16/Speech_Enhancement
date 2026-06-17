"""
Weight conversion: PyTorch checkpoint → TensorFlow/Keras .h5 file.

WHY DIRECT LOADING IS NOT POSSIBLE:
    PyTorch and TensorFlow store convolution kernels in different axis orders.
    You cannot load a .pt file directly into a Keras model — the shapes do not
    match and the values would be scrambled.

    PyTorch Conv2d kernel:          [out_channels, in_channels, H, W]
    TensorFlow Conv2D kernel:       [H, W, in_channels, out_channels]
    Required transpose:             np.transpose(w, (2, 3, 1, 0))

    PyTorch ConvTranspose2d kernel: [in_channels, out_channels, H, W]
    TensorFlow Conv2DTranspose:     [H, W, out_channels, in_channels]
    Required transpose:             np.transpose(w, (2, 3, 1, 0))  ← same formula

    BatchNormalization weights need no reordering — they are 1-D vectors of
    length equal to the number of channels.

REQUIREMENTS (for this script only):
    pip install torch          # to read the .pt checkpoint
    pip install tensorflow     # to build the Keras model and save .h5

    PyTorch is NOT needed for training, inference, or evaluation after conversion.

USAGE:
    python weights/convert_weights.py

OUTPUT:
    models/unet_tf_weights.h5   ← Keras weight file, used by train.py and enhance.py
"""

import sys
import numpy as np

# ── Verify dependencies before doing anything else ────────────────────────────
try:
    import torch
except ImportError:
    sys.exit("PyTorch is required for weight conversion: pip install torch")

try:
    import tensorflow as tf
except ImportError:
    sys.exit("TensorFlow is required: pip install tensorflow")

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet import build_unet

# ── Paths ─────────────────────────────────────────────────────────────────────
PT_CHECKPOINT = "models/unet_final_continued.pt"
TF_WEIGHTS    = "models/unet_tf_weights.h5"


# ── Helper functions ───────────────────────────────────────────────────────────

def to_numpy(pt_tensor):
    """Detach a PyTorch tensor and return it as a NumPy array."""
    return pt_tensor.detach().cpu().numpy()


def transpose_conv_kernel(pt_weight):
    """Reorder a Conv2d kernel from PyTorch layout to TensorFlow layout.

    PyTorch: [out_channels, in_channels, H, W]
    TF:      [H, W, in_channels, out_channels]
    """
    return np.transpose(to_numpy(pt_weight), (2, 3, 1, 0))


def transpose_deconv_kernel(pt_weight):
    """Reorder a ConvTranspose2d kernel from PyTorch layout to TensorFlow layout.

    PyTorch: [in_channels, out_channels, H, W]
    TF:      [H, W, out_channels, in_channels]
    """
    return np.transpose(to_numpy(pt_weight), (2, 3, 1, 0))


def set_conv_weights(tf_layer, sd, pt_prefix):
    """Copy Conv2d weights from a PyTorch state_dict into a TF Conv2D layer."""
    kernel = transpose_conv_kernel(sd[f"{pt_prefix}.weight"])
    bias   = to_numpy(sd[f"{pt_prefix}.bias"])
    tf_layer.set_weights([kernel, bias])


def set_deconv_weights(tf_layer, sd, pt_prefix):
    """Copy ConvTranspose2d weights into a TF Conv2DTranspose layer."""
    kernel = transpose_deconv_kernel(sd[f"{pt_prefix}.weight"])
    bias   = to_numpy(sd[f"{pt_prefix}.bias"])
    tf_layer.set_weights([kernel, bias])


def set_batchnorm_weights(tf_layer, sd, pt_prefix):
    """Copy BatchNorm2d weights into a TF BatchNormalization layer.

    PyTorch names:     weight (γ), bias (β), running_mean, running_var
    TF set_weights order: [γ, β, moving_mean, moving_variance]
    """
    gamma        = to_numpy(sd[f"{pt_prefix}.weight"])
    beta         = to_numpy(sd[f"{pt_prefix}.bias"])
    moving_mean  = to_numpy(sd[f"{pt_prefix}.running_mean"])
    moving_var   = to_numpy(sd[f"{pt_prefix}.running_var"])
    tf_layer.set_weights([gamma, beta, moving_mean, moving_var])


def set_convblock_weights(tf_block, sd, pt_prefix):
    """Copy a full ConvBlock (2× Conv + BN pairs) from PyTorch to TF.

    PyTorch Sequential indices:
        0 → Conv2d      maps to tf_block.conv1
        1 → BatchNorm2d maps to tf_block.bn1
        3 → Conv2d      maps to tf_block.conv2
        4 → BatchNorm2d maps to tf_block.bn2
        (indices 2 and 5 are ReLU, no parameters)
    """
    set_conv_weights(     tf_block.conv1, sd, f"{pt_prefix}.block.0")
    set_batchnorm_weights(tf_block.bn1,   sd, f"{pt_prefix}.block.1")
    set_conv_weights(     tf_block.conv2, sd, f"{pt_prefix}.block.3")
    set_batchnorm_weights(tf_block.bn2,   sd, f"{pt_prefix}.block.4")


# ── Main conversion ────────────────────────────────────────────────────────────

def convert(pt_checkpoint_path, tf_weights_path):
    print("=" * 60)
    print("PyTorch → TensorFlow Weight Conversion")
    print("=" * 60)

    # 1. Load PyTorch checkpoint
    if not os.path.exists(pt_checkpoint_path):
        sys.exit(f"Checkpoint not found: {pt_checkpoint_path}")

    checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")
    sd = checkpoint["model_state_dict"]
    print(f"  Loaded PyTorch checkpoint (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Loss at save time: {checkpoint.get('loss', 'N/A')}")

    # 2. Build TF model and force weight initialization via a dummy forward pass
    model = build_unet()
    dummy = tf.zeros([1, 257, 501, 1], dtype=tf.float32)
    model(dummy, training=False)
    print(f"\n  TF model built: {model.count_params():,} parameters")

    # 3. Copy encoder weights
    print("\n  Copying encoder blocks...")
    set_convblock_weights(model.encoder_block1, sd, "enc1")
    set_convblock_weights(model.encoder_block2, sd, "enc2")
    set_convblock_weights(model.encoder_block3, sd, "enc3")
    set_convblock_weights(model.encoder_block4, sd, "enc4")
    print("    encoder_block1 (1→16)   ✓")
    print("    encoder_block2 (16→32)  ✓")
    print("    encoder_block3 (32→64)  ✓")
    print("    encoder_block4 (64→128) ✓")

    # 4. Copy bottleneck weights
    print("\n  Copying bottleneck...")
    set_convblock_weights(model.bottleneck, sd, "bottleneck")
    print("    bottleneck (128→256) ✓")

    # 5. Copy upsampling (ConvTranspose2d) weights
    print("\n  Copying upsample layers...")
    set_deconv_weights(model.upsample4, sd, "up4")
    set_deconv_weights(model.upsample3, sd, "up3")
    set_deconv_weights(model.upsample2, sd, "up2")
    set_deconv_weights(model.upsample1, sd, "up1")
    print("    upsample4 (256→128) ✓")
    print("    upsample3 (128→64)  ✓")
    print("    upsample2 (64→32)   ✓")
    print("    upsample1 (32→16)   ✓")

    # 6. Copy decoder weights
    print("\n  Copying decoder blocks...")
    set_convblock_weights(model.decoder_block4, sd, "dec4")
    set_convblock_weights(model.decoder_block3, sd, "dec3")
    set_convblock_weights(model.decoder_block2, sd, "dec2")
    set_convblock_weights(model.decoder_block1, sd, "dec1")
    print("    decoder_block4 (256→128) ✓")
    print("    decoder_block3 (128→64)  ✓")
    print("    decoder_block2 (64→32)   ✓")
    print("    decoder_block1 (32→16)   ✓")

    # 7. Copy final 1×1 conv
    print("\n  Copying output conv...")
    set_conv_weights(model.output_conv, sd, "final")
    print("    output_conv (16→1) ✓")

    # 8. Verify shapes match by running another forward pass and checking output
    print("\n  Running sanity check...")
    test_input  = tf.random.uniform([1, 257, 501, 1])
    test_output = model(test_input, training=False)
    assert test_output.shape == (1, 257, 501, 1), \
        f"Unexpected output shape: {test_output.shape}"
    print(f"    Input:  {list(test_input.shape)}")
    print(f"    Output: {list(test_output.shape)}  ✓")

    # 9. Save TF weights
    os.makedirs(os.path.dirname(tf_weights_path), exist_ok=True)
    model.save_weights(tf_weights_path)
    print(f"\n  Saved TF weights → {tf_weights_path}")
    print("\n" + "=" * 60)
    print("Conversion complete. You can now use models/unet_tf_weights.h5")
    print("PyTorch is no longer needed for inference or training.")
    print("=" * 60)


if __name__ == "__main__":
    convert(PT_CHECKPOINT, TF_WEIGHTS)
