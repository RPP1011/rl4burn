//! Neural network building blocks for reinforcement learning with Burn.
//!
//! This crate provides initialization utilities, gradient clipping, attention
//! mechanisms, recurrent cells, variational autoencoders, and other modules
//! commonly needed when building RL agents on top of the Burn framework.

pub mod attention;
pub mod autoregressive;
pub mod clip;
pub mod conv;
pub mod dist;
pub mod film;
pub mod init;
pub mod mlp;
pub mod multi_encoder;
pub mod policy;
pub mod polyak;
pub mod rnn;
pub mod rssm;
pub mod symlog;
pub mod vae;

pub use attention::{
    AttentionPool, AttentionPoolConfig, MultiHeadAttention, MultiHeadAttentionConfig, PointerNet,
    PointerNetConfig, TargetAttention, TargetAttentionConfig, TransformerBlock,
    TransformerBlockConfig, TransformerEncoder, TransformerEncoderConfig,
};
pub use autoregressive::{ActionHead, CompositeDistribution};
pub use clip::clip_grad_norm;
pub use dist::{ActionDist, LogStdMode};
pub use film::{Film, FilmConfig};
pub use init::orthogonal_linear;
pub use policy::{DiscreteAcOutput, DiscreteActorCritic, greedy_action};
pub use polyak::polyak_update;
pub use rnn::{
    BlockGruCell, BlockGruCellConfig, GruCell, GruCellConfig, LstmCell, LstmCellConfig, LstmState,
};
pub use conv::{ConvDecoder, ConvDecoderConfig, ConvEncoder, ConvEncoderConfig};
pub use mlp::{Mlp, MlpConfig, NormKind, RmsNorm, RmsNormConfig};
pub use multi_encoder::{MultiDecoder, MultiDecoderConfig, MultiEncoder, MultiEncoderConfig};
pub use rssm::{Rssm, RssmConfig, RssmState};
pub use symlog::{TwohotEncoder, symexp, symlog};
pub use vae::{BetaVae, BetaVaeConfig, VaeOutput};
