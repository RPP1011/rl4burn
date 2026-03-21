//! Multi-modal encoder and decoder for mixed observation spaces.
//!
//! Routes image observations through a CNN encoder and vector observations
//! through an MLP, then concatenates the results.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::conv::{ConvDecoder, ConvDecoderConfig, ConvEncoder, ConvEncoderConfig};
use crate::mlp::{Mlp, MlpConfig, NormKind};

// ---------------------------------------------------------------------------
// MultiEncoder
// ---------------------------------------------------------------------------

/// Configuration for [`MultiEncoder`].
#[derive(Config, Debug)]
pub struct MultiEncoderConfig {
    /// Dimension of vector observations (0 if image-only).
    #[config(default = 0)]
    pub vector_dim: usize,
    /// Image encoder config (None if vector-only).
    pub image_config: Option<ConvEncoderConfig>,
    /// Hidden size for the vector observation MLP.
    #[config(default = 256)]
    pub vector_hidden: usize,
    /// Output embedding dimension.
    pub embed_dim: usize,
}

/// Encodes mixed observations (images + vectors) into a unified embedding.
#[derive(Module, Debug)]
pub struct MultiEncoder<B: Backend> {
    image_encoder: Option<ConvEncoder<B>>,
    vector_mlp: Option<Mlp<B>>,
    fuse: Linear<B>,
    #[module(skip)]
    embed_dim: usize,
}

impl MultiEncoderConfig {
    /// Initialize a [`MultiEncoder`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiEncoder<B> {
        let image_encoder = self.image_config.as_ref().map(|c| c.init(device));
        let image_dim = self.image_config.as_ref().map_or(0, |c| c.embed_dim());

        let vector_mlp = if self.vector_dim > 0 {
            Some(
                MlpConfig::new(self.vector_dim, self.vector_hidden, self.vector_hidden)
                    .with_n_layers(1)
                    .init_with_norm(NormKind::Rms, device),
            )
        } else {
            None
        };
        let vector_out = if self.vector_dim > 0 {
            self.vector_hidden
        } else {
            0
        };

        let fuse_input = image_dim + vector_out;
        let fuse = LinearConfig::new(fuse_input, self.embed_dim).init(device);

        MultiEncoder {
            image_encoder,
            vector_mlp,
            fuse,
            embed_dim: self.embed_dim,
        }
    }
}

impl<B: Backend> MultiEncoder<B> {
    /// Encode observations.
    ///
    /// * `image` — optional `[batch, C, H, W]` image tensor.
    /// * `vector` — optional `[batch, vector_dim]` vector tensor.
    ///
    /// Returns `[batch, embed_dim]`.
    pub fn forward(
        &self,
        image: Option<Tensor<B, 4>>,
        vector: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let mut parts: Vec<Tensor<B, 2>> = Vec::new();

        if let (Some(enc), Some(img)) = (&self.image_encoder, image) {
            parts.push(enc.forward(img));
        }
        if let (Some(mlp), Some(vec)) = (&self.vector_mlp, vector) {
            parts.push(mlp.forward(vec));
        }

        let fused = if parts.len() == 1 {
            parts.remove(0)
        } else {
            Tensor::cat(parts, 1)
        };

        self.fuse.forward(fused)
    }

    /// Output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ---------------------------------------------------------------------------
// MultiDecoder
// ---------------------------------------------------------------------------

/// Configuration for [`MultiDecoder`].
#[derive(Config, Debug)]
pub struct MultiDecoderConfig {
    /// Dimension of vector observations to reconstruct (0 if image-only).
    #[config(default = 0)]
    pub vector_dim: usize,
    /// Image decoder config (None if vector-only).
    pub image_config: Option<ConvDecoderConfig>,
    /// Input latent dimension.
    pub input_dim: usize,
}

/// Decodes latent features into mixed observations.
#[derive(Module, Debug)]
pub struct MultiDecoder<B: Backend> {
    image_decoder: Option<ConvDecoder<B>>,
    vector_head: Option<Linear<B>>,
}

impl MultiDecoderConfig {
    /// Initialize a [`MultiDecoder`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiDecoder<B> {
        let image_decoder = self.image_config.as_ref().map(|c| c.init(device));
        let vector_head = if self.vector_dim > 0 {
            Some(LinearConfig::new(self.input_dim, self.vector_dim).init(device))
        } else {
            None
        };

        MultiDecoder {
            image_decoder,
            vector_head,
        }
    }
}

impl<B: Backend> MultiDecoder<B> {
    /// Decode latent features.
    ///
    /// Returns `(image, vector)` — each is `None` if the corresponding
    /// modality is not configured.
    pub fn forward(
        &self,
        latent: Tensor<B, 2>,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 2>>) {
        let image = self
            .image_decoder
            .as_ref()
            .map(|dec| dec.forward(latent.clone()));
        let vector = self
            .vector_head
            .as_ref()
            .map(|head| head.forward(latent));

        (image, vector)
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
    fn multi_encoder_vector_only() {
        let config = MultiEncoderConfig::new(64).with_vector_dim(8);
        let enc = config.init::<B>(&dev());
        let out = enc.forward(None, Some(Tensor::zeros([2, 8], &dev())));
        assert_eq!(out.dims(), [2, 64]);
    }

    #[test]
    fn multi_encoder_image_only() {
        let img_cfg = ConvEncoderConfig::new()
            .with_channel_mult(8)
            .with_depth(3)
            .with_image_size(32)
            .with_kernel_size(4);
        let config = MultiEncoderConfig::new(64).with_image_config(Some(img_cfg));
        let enc = config.init::<B>(&dev());
        let img = Tensor::<B, 4>::zeros([2, 3, 32, 32], &dev());
        let out = enc.forward(Some(img), None);
        assert_eq!(out.dims(), [2, 64]);
    }

    #[test]
    fn multi_decoder_vector() {
        let config = MultiDecoderConfig::new(32).with_vector_dim(8);
        let dec = config.init::<B>(&dev());
        let (img, vec) = dec.forward(Tensor::zeros([2, 32], &dev()));
        assert!(img.is_none());
        assert_eq!(vec.unwrap().dims(), [2, 8]);
    }
}
