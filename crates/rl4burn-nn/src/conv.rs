//! CNN encoder and decoder for image observations (R2-Dreamer style).
//!
//! * [`ConvEncoder`] — Stacks of stride-2 Conv2d layers that map
//!   `[B, C, H, W]` images to a flat feature vector `[B, embed_dim]`.
//! * [`ConvDecoder`] — Inverse of the encoder using ConvTranspose2d layers.
//!
//! Channel progression follows R2-Dreamer: multiplier × {1, 2, 4, 8}
//! (e.g. 48, 96, 192, 384 for multiplier=48).

use burn::nn::conv::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;

use crate::mlp::{RmsNorm, RmsNormConfig};

// ---------------------------------------------------------------------------
// ConvEncoder
// ---------------------------------------------------------------------------

/// Configuration for [`ConvEncoder`].
#[derive(Config, Debug)]
pub struct ConvEncoderConfig {
    /// Number of input image channels (e.g. 3 for RGB, 1 for grayscale).
    #[config(default = 3)]
    pub in_channels: usize,
    /// Base channel multiplier. Layers use mult × {1, 2, 4, 8}.
    #[config(default = 48)]
    pub channel_mult: usize,
    /// Number of convolutional layers (default 4).
    #[config(default = 4)]
    pub depth: usize,
    /// Kernel size for all conv layers.
    #[config(default = 4)]
    pub kernel_size: usize,
    /// Expected spatial dimension of input (H = W). Used to compute
    /// the flattened feature size. Default: 64.
    #[config(default = 64)]
    pub image_size: usize,
}

impl ConvEncoderConfig {
    /// Channel count at layer `i`: `channel_mult * 2^i`.
    fn channels_at(&self, i: usize) -> usize {
        self.channel_mult * (1 << i.min(self.depth - 1))
    }

    /// Output embedding dimension after flattening.
    pub fn embed_dim(&self) -> usize {
        let spatial = self.image_size >> self.depth; // each layer halves
        let last_ch = self.channels_at(self.depth - 1);
        last_ch * spatial * spatial
    }
}

/// CNN encoder: images → flat features.
#[derive(Module, Debug)]
pub struct ConvEncoder<B: Backend> {
    convs: Vec<Conv2d<B>>,
    norms: Vec<RmsNorm<B>>,
    #[module(skip)]
    embed_dim: usize,
}

impl ConvEncoderConfig {
    /// Initialize a [`ConvEncoder`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvEncoder<B> {
        let mut convs = Vec::with_capacity(self.depth);
        let mut norms = Vec::with_capacity(self.depth);

        for i in 0..self.depth {
            let in_ch = if i == 0 {
                self.in_channels
            } else {
                self.channels_at(i - 1)
            };
            let out_ch = self.channels_at(i);

            // Stride-2, kernel 4, same-ish padding (pad=1 with kernel 4, stride 2
            // gives exact halving: out = (in + 2*1 - 4)/2 + 1 = in/2).
            convs.push(
                Conv2dConfig::new([in_ch, out_ch], [self.kernel_size, self.kernel_size])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(
                        (self.kernel_size / 2 - 1).max(0),
                        (self.kernel_size / 2 - 1).max(0),
                    ))
                    .with_bias(false)
                    .init(device),
            );
            // Flatten for norm: treat channels as features
            norms.push(RmsNormConfig::new(out_ch).init(device));
        }

        ConvEncoder {
            convs,
            norms,
            embed_dim: self.embed_dim(),
        }
    }
}

impl<B: Backend> ConvEncoder<B> {
    /// Encode an image batch to a flat feature vector.
    ///
    /// Input: `[batch, C, H, W]`. Output: `[batch, embed_dim]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut h = x;
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            h = conv.forward(h);
            // Apply RmsNorm over channels: reshape to [B, C, H*W], norm over C,
            // then reshape back.
            let [b, c, height, w] = h.dims();
            // Permute to [B, H*W, C], norm, permute back.
            let flat = h.swap_dims(1, 3).reshape([b * height * w, c]);
            let normed = norm.forward(flat);
            h = normed.reshape([b, w, height, c]).swap_dims(1, 3);
            h = burn::tensor::activation::silu(h);
        }
        let [batch, _, _, _] = h.dims();
        h.reshape([batch, self.embed_dim])
    }

    /// Output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ---------------------------------------------------------------------------
// ConvDecoder
// ---------------------------------------------------------------------------

/// Configuration for [`ConvDecoder`].
#[derive(Config, Debug)]
pub struct ConvDecoderConfig {
    /// Number of output image channels.
    #[config(default = 3)]
    pub out_channels: usize,
    /// Base channel multiplier (same as encoder).
    #[config(default = 48)]
    pub channel_mult: usize,
    /// Number of transposed-conv layers (should match encoder depth).
    #[config(default = 4)]
    pub depth: usize,
    /// Kernel size.
    #[config(default = 4)]
    pub kernel_size: usize,
    /// Spatial dim of the initial feature map before upsampling.
    /// For image_size=64 and depth=4, this is 64/16 = 4.
    #[config(default = 4)]
    pub init_spatial: usize,
    /// Input feature dimension (from latent state).
    pub input_dim: usize,
}

impl ConvDecoderConfig {
    fn channels_at(&self, i: usize) -> usize {
        self.channel_mult * (1 << i.min(self.depth - 1))
    }
}

/// CNN decoder: flat features → images.
#[derive(Module, Debug)]
pub struct ConvDecoder<B: Backend> {
    proj: burn::nn::Linear<B>,
    deconvs: Vec<ConvTranspose2d<B>>,
    norms: Vec<RmsNorm<B>>,
    #[module(skip)]
    init_spatial: usize,
    #[module(skip)]
    init_channels: usize,
}

impl ConvDecoderConfig {
    /// Initialize a [`ConvDecoder`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvDecoder<B> {
        let init_channels = self.channels_at(self.depth - 1);
        let proj_dim = init_channels * self.init_spatial * self.init_spatial;
        let proj = burn::nn::LinearConfig::new(self.input_dim, proj_dim).init(device);

        let mut deconvs = Vec::with_capacity(self.depth);
        let mut norms = Vec::with_capacity(self.depth);

        // Decode in reverse order: highest channels first
        for i in (0..self.depth).rev() {
            let in_ch = self.channels_at(i);
            let out_ch = if i == 0 {
                self.out_channels
            } else {
                self.channels_at(i - 1)
            };

            let padding = (self.kernel_size / 2 - 1).max(0);
            deconvs.push(
                ConvTranspose2dConfig::new([in_ch, out_ch], [self.kernel_size, self.kernel_size])
                    .with_stride([2, 2])
                    .with_padding([padding, padding])
                    .with_bias(false)
                    .init(device),
            );
            // No norm on last layer
            if i > 0 {
                norms.push(RmsNormConfig::new(out_ch).init(device));
            }
        }

        ConvDecoder {
            proj,
            deconvs,
            norms,
            init_spatial: self.init_spatial,
            init_channels,
        }
    }
}

impl<B: Backend> ConvDecoder<B> {
    /// Decode a flat feature vector to images.
    ///
    /// Input: `[batch, input_dim]`. Output: `[batch, out_channels, H, W]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, _] = x.dims();
        let h = self.proj.forward(x);
        let mut h = h.reshape([batch, self.init_channels, self.init_spatial, self.init_spatial]);

        for (i, deconv) in self.deconvs.iter().enumerate() {
            h = deconv.forward(h);
            // Apply norm + activation for all but last layer
            if i < self.norms.len() {
                let [b, c, height, w] = h.dims();
                let flat = h.swap_dims(1, 3).reshape([b * height * w, c]);
                let normed = self.norms[i].forward(flat);
                h = normed.reshape([b, w, height, c]).swap_dims(1, 3);
                h = burn::tensor::activation::silu(h);
            }
        }

        h
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn encoder_output_shape() {
        let config = ConvEncoderConfig::new()
            .with_in_channels(3)
            .with_channel_mult(16)
            .with_depth(4)
            .with_kernel_size(4)
            .with_image_size(64);
        let encoder = config.init::<B>(&dev());
        let x = Tensor::<B, 4>::zeros([2, 3, 64, 64], &dev());
        let y = encoder.forward(x);
        // 64 >> 4 = 4, last channel = 16 * 8 = 128, embed = 128 * 4 * 4 = 2048
        assert_eq!(y.dims(), [2, config.embed_dim()]);
    }

    #[test]
    fn encoder_embed_dim_calculation() {
        let config = ConvEncoderConfig::new()
            .with_channel_mult(48)
            .with_depth(4)
            .with_image_size(64);
        // 64 >> 4 = 4, last_ch = 48 * 8 = 384, embed = 384 * 4 * 4 = 6144
        assert_eq!(config.embed_dim(), 384 * 4 * 4);
    }

    #[test]
    fn decoder_output_shape() {
        let config = ConvDecoderConfig::new(256)
            .with_out_channels(3)
            .with_channel_mult(16)
            .with_depth(4)
            .with_kernel_size(4)
            .with_init_spatial(4);
        let decoder = config.init::<B>(&dev());
        let x = Tensor::<B, 2>::zeros([2, 256], &dev());
        let y = decoder.forward(x);
        assert_eq!(y.dims()[0], 2);
        assert_eq!(y.dims()[1], 3);
        assert_eq!(y.dims()[2], 64); // 4 * 2^4 = 64
        assert_eq!(y.dims()[3], 64);
    }

    #[test]
    fn encoder_values_finite() {
        let config = ConvEncoderConfig::new()
            .with_channel_mult(8)
            .with_depth(3)
            .with_image_size(32)
            .with_kernel_size(4);
        let encoder = config.init::<B>(&dev());
        let x = Tensor::<B, 4>::random(
            [2, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let y = encoder.forward(x);
        let vals: Vec<f32> = y.to_data().to_vec().unwrap();
        for &v in &vals {
            assert!(v.is_finite(), "encoder output should be finite, got {v}");
        }
    }

    #[test]
    fn decoder_values_finite() {
        let config = ConvDecoderConfig::new(64)
            .with_channel_mult(8)
            .with_depth(3)
            .with_kernel_size(4)
            .with_init_spatial(4);
        let decoder = config.init::<B>(&dev());
        let x = Tensor::<B, 2>::random(
            [2, 64],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let y = decoder.forward(x);
        let vals: Vec<f32> = y.to_data().to_vec().unwrap();
        for &v in &vals {
            assert!(v.is_finite(), "decoder output should be finite, got {v}");
        }
    }
}
