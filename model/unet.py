"""
U-Net architecture for speech enhancement.

The model operates on magnitude spectrograms (2D images) and learns to predict
the clean magnitude given a noisy one. Skip connections from encoder to decoder
help preserve fine-grained frequency detail that would otherwise be lost in the
pooling layers.

Input shape:  [batch, freq_bins, time_frames, 1]  (channels-last, single channel)
Output shape: [batch, freq_bins, time_frames, 1]  (predicted clean magnitude)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization
)


class ConvBlock(tf.keras.layers.Layer):
    """Two consecutive Conv → BatchNorm → ReLU operations.

    This is the fundamental building block of the U-Net. Used in both
    the encoder and decoder paths.
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters, kernel_size=3, padding="same", use_bias=True)
        self.bn1   = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size=3, padding="same", use_bias=True)
        self.bn2   = BatchNormalization()

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        return x


class UNet(tf.keras.Model):
    """U-Net for speech enhancement.

    Encoder compresses the spectrogram through 4 stages, each halving the
    spatial dimensions while doubling the channel count. The bottleneck
    learns the most abstract representation. The decoder mirrors the encoder,
    upsampling at each stage and concatenating the skip connection from the
    matching encoder stage to recover spatial detail.

    Architecture:
        Encoder:      1 → 16 → 32 → 64 → 128 channels
        Bottleneck:   128 → 256 channels
        Decoder:      256 → 128 → 64 → 32 → 16 → 1 channels
    """

    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.encoder_block1 = ConvBlock(16,  name="encoder_block1")
        self.encoder_block2 = ConvBlock(32,  name="encoder_block2")
        self.encoder_block3 = ConvBlock(64,  name="encoder_block3")
        self.encoder_block4 = ConvBlock(128, name="encoder_block4")
        self.pool = MaxPooling2D(pool_size=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(256, name="bottleneck")

        # --- Decoder (upsample + ConvBlock at each stage) ---
        # Each upsample layer doubles spatial dimensions and halves channels
        self.upsample4 = Conv2DTranspose(128, kernel_size=2, strides=2, name="upsample4")
        self.upsample3 = Conv2DTranspose(64,  kernel_size=2, strides=2, name="upsample3")
        self.upsample2 = Conv2DTranspose(32,  kernel_size=2, strides=2, name="upsample2")
        self.upsample1 = Conv2DTranspose(16,  kernel_size=2, strides=2, name="upsample1")

        # After upsampling, channels double due to skip connection concat
        # e.g. upsample4 gives 128ch, concat with e4 (128ch) → 256ch input for dec4
        self.decoder_block4 = ConvBlock(128, name="decoder_block4")
        self.decoder_block3 = ConvBlock(64,  name="decoder_block3")
        self.decoder_block2 = ConvBlock(32,  name="decoder_block2")
        self.decoder_block1 = ConvBlock(16,  name="decoder_block1")

        # 1×1 conv to collapse 16 channels down to 1 (the enhanced magnitude)
        self.output_conv = Conv2D(1, kernel_size=1, name="output_conv")

    def call(self, inputs, training=False):
        # ---- Encoder path (compress + save skip connections) ----
        e1 = self.encoder_block1(inputs,         training=training)
        e2 = self.encoder_block2(self.pool(e1),  training=training)
        e3 = self.encoder_block3(self.pool(e2),  training=training)
        e4 = self.encoder_block4(self.pool(e3),  training=training)

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool(e4), training=training)

        # ---- Decoder path (expand + fuse skip connections) ----
        d4 = self._upsample_and_fuse(self.upsample4, self.decoder_block4, b,  e4, training)
        d3 = self._upsample_and_fuse(self.upsample3, self.decoder_block3, d4, e3, training)
        d2 = self._upsample_and_fuse(self.upsample2, self.decoder_block2, d3, e2, training)
        d1 = self._upsample_and_fuse(self.upsample1, self.decoder_block1, d2, e1, training)

        return self.output_conv(d1)

    def _upsample_and_fuse(self, upsample_layer, conv_block, x, skip, training):
        """Upsample x, align with skip connection, concatenate, then convolve."""
        x = upsample_layer(x)
        x = self._match_size(x, skip)
        x = tf.concat([x, skip], axis=-1)  # concatenate along channel axis
        return conv_block(x, training=training)

    @staticmethod
    def _match_size(x, target):
        """Crop x to match target's spatial dimensions (freq and time axes).

        Slight mismatches can occur after transposed convolution when input
        dimensions are odd. Cropping is cheaper and more stable than padding.
        """
        target_h = tf.shape(target)[1]
        target_w = tf.shape(target)[2]
        return x[:, :target_h, :target_w, :]


def build_unet():
    """Construct and return the U-Net model."""
    return UNet()
