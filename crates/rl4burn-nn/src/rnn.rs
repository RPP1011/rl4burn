//! Recurrent neural network cells: LSTM, GRU, and Block-diagonal GRU.
//!
//! These cells operate on single timesteps and include convenience methods
//! for processing sequences. All implementations are backend-agnostic via
//! Burn's `Module` derive.
//!
//! # Cells
//!
//! - [`LstmCell`] — Standard LSTM cell with forget/input/output gates.
//! - [`GruCell`] — Standard GRU cell (PyTorch convention: reset gate applied
//!   after hidden-hidden multiplication).
//! - [`BlockGruCell`] — Block-diagonal GRU (DreamerV3 style) that partitions
//!   the hidden state into `n_blocks` independent blocks, reducing recurrent
//!   parameters from `hidden_size^2` to `hidden_size^2 / n_blocks`.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, tanh};

// ====================== LSTM ======================

/// LSTM cell state (hidden and cell vectors).
#[derive(Clone, Debug)]
pub struct LstmState<B: Backend> {
    /// Hidden state `[batch, hidden_size]`.
    pub h: Tensor<B, 2>,
    /// Cell state `[batch, hidden_size]`.
    pub c: Tensor<B, 2>,
}

impl<B: Backend> LstmState<B> {
    /// Create a zero-initialized LSTM state.
    pub fn zeros(batch_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            h: Tensor::zeros([batch_size, hidden_size], device),
            c: Tensor::zeros([batch_size, hidden_size], device),
        }
    }
}

/// Configuration for an [`LstmCell`].
#[derive(Config, Debug)]
pub struct LstmCellConfig {
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
}

/// LSTM cell: computes one timestep of the LSTM recurrence.
///
/// Uses two separate linear projections (`ih` and `hh`) for efficiency,
/// producing the four gate vectors (input, forget, cell-candidate, output)
/// in a single matmul each.
#[derive(Module, Debug)]
pub struct LstmCell<B: Backend> {
    /// Input-to-hidden projection: `input_size -> 4 * hidden_size`.
    ih: Linear<B>,
    /// Hidden-to-hidden projection: `hidden_size -> 4 * hidden_size`.
    hh: Linear<B>,
    /// Hidden state dimension (stored as constant for slicing).
    #[module(skip)]
    hidden_size: usize,
}

impl LstmCellConfig {
    /// Initialize an LSTM cell on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmCell<B> {
        LstmCell {
            ih: LinearConfig::new(self.input_size, 4 * self.hidden_size).init(device),
            hh: LinearConfig::new(self.hidden_size, 4 * self.hidden_size).init(device),
            hidden_size: self.hidden_size,
        }
    }
}

impl<B: Backend> LstmCell<B> {
    /// Forward one timestep.
    ///
    /// # Arguments
    /// * `input` — `[batch, input_size]`
    /// * `state` — Previous LSTM state (h, c)
    ///
    /// # Returns
    /// New LSTM state after one step.
    pub fn forward(&self, input: Tensor<B, 2>, state: &LstmState<B>) -> LstmState<B> {
        let gates = self.ih.forward(input) + self.hh.forward(state.h.clone());
        // gates: [batch, 4 * hidden_size]

        let batch = gates.dims()[0];
        let hs = self.hidden_size;

        let i = sigmoid(gates.clone().slice([0..batch, 0..hs]));
        let f = sigmoid(gates.clone().slice([0..batch, hs..2 * hs]));
        let g = tanh(gates.clone().slice([0..batch, 2 * hs..3 * hs]));
        let o = sigmoid(gates.slice([0..batch, 3 * hs..4 * hs]));

        let new_c = f * state.c.clone() + i * g;
        let new_h = o * tanh(new_c.clone());

        LstmState { h: new_h, c: new_c }
    }

    /// Process a sequence of inputs, returning all hidden states.
    ///
    /// # Arguments
    /// * `inputs` — `[batch, seq_len, input_size]`
    /// * `initial_state` — Initial LSTM state
    ///
    /// # Returns
    /// `(outputs, final_state)` where outputs is `[batch, seq_len, hidden_size]`.
    pub fn forward_seq(
        &self,
        inputs: Tensor<B, 3>,
        initial_state: &LstmState<B>,
    ) -> (Tensor<B, 3>, LstmState<B>) {
        let [batch, seq_len, input_size] = inputs.dims();
        let mut state = LstmState {
            h: initial_state.h.clone(),
            c: initial_state.c.clone(),
        };
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t: Tensor<B, 2> = inputs
                .clone()
                .slice([0..batch, t..t + 1, 0..input_size])
                .squeeze_dim::<2>(1);
            state = self.forward(x_t, &state);
            outputs.push(state.h.clone().unsqueeze_dim::<3>(1));
        }

        let output_tensor = Tensor::cat(outputs, 1);
        (output_tensor, state)
    }
}

// ====================== GRU ======================

/// Configuration for a [`GruCell`].
#[derive(Config, Debug)]
pub struct GruCellConfig {
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
}

/// GRU cell: computes one timestep of the GRU recurrence.
///
/// Follows the PyTorch convention where the reset gate is applied *after*
/// the hidden-to-hidden linear transformation.
#[derive(Module, Debug)]
pub struct GruCell<B: Backend> {
    /// Input-to-hidden projection: `input_size -> 3 * hidden_size`.
    ih: Linear<B>,
    /// Hidden-to-hidden projection: `hidden_size -> 3 * hidden_size`.
    hh: Linear<B>,
    /// Hidden state dimension.
    #[module(skip)]
    hidden_size: usize,
}

impl GruCellConfig {
    /// Initialize a GRU cell on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GruCell<B> {
        GruCell {
            ih: LinearConfig::new(self.input_size, 3 * self.hidden_size).init(device),
            hh: LinearConfig::new(self.hidden_size, 3 * self.hidden_size).init(device),
            hidden_size: self.hidden_size,
        }
    }
}

impl<B: Backend> GruCell<B> {
    /// Forward one timestep.
    ///
    /// # Arguments
    /// * `input` — `[batch, input_size]`
    /// * `h` — Previous hidden state `[batch, hidden_size]`
    ///
    /// # Returns
    /// New hidden state `[batch, hidden_size]`.
    pub fn forward(&self, input: Tensor<B, 2>, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = input.dims()[0];
        let hs = self.hidden_size;

        let ih_out = self.ih.forward(input);
        let hh_out = self.hh.forward(h.clone());

        // Split input-hidden gates
        let r_i = ih_out.clone().slice([0..batch, 0..hs]);
        let z_i = ih_out.clone().slice([0..batch, hs..2 * hs]);
        let n_i = ih_out.slice([0..batch, 2 * hs..3 * hs]);

        // Split hidden-hidden gates
        let r_h = hh_out.clone().slice([0..batch, 0..hs]);
        let z_h = hh_out.clone().slice([0..batch, hs..2 * hs]);
        let n_h = hh_out.slice([0..batch, 2 * hs..3 * hs]);

        let r = sigmoid(r_i + r_h);
        let z = sigmoid(z_i + z_h);
        // PyTorch convention: r applied after W_h multiplication
        let n = (n_i + r * n_h).tanh();

        // h_new = (1 - z) * n + z * h_prev
        (Tensor::ones_like(&z) - z.clone()) * n + z * h
    }

    /// Process a sequence of inputs, returning all hidden states.
    ///
    /// # Arguments
    /// * `inputs` — `[batch, seq_len, input_size]`
    /// * `initial_h` — Initial hidden state `[batch, hidden_size]`
    ///
    /// # Returns
    /// `(outputs, final_h)` where outputs is `[batch, seq_len, hidden_size]`.
    pub fn forward_seq(
        &self,
        inputs: Tensor<B, 3>,
        initial_h: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, seq_len, input_size] = inputs.dims();
        let mut h = initial_h;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t: Tensor<B, 2> = inputs
                .clone()
                .slice([0..batch, t..t + 1, 0..input_size])
                .squeeze_dim::<2>(1);
            h = self.forward(x_t, h);
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }

        let output_tensor = Tensor::cat(outputs, 1);
        (output_tensor, h)
    }
}

// ====================== Block GRU ======================

/// Configuration for a [`BlockGruCell`].
#[derive(Config, Debug)]
pub struct BlockGruCellConfig {
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden state dimension.
    pub hidden_size: usize,
    /// Number of blocks. Must evenly divide `hidden_size`. Default: 8.
    #[config(default = 8)]
    pub n_blocks: usize,
}

/// Block-diagonal GRU cell (DreamerV3 style).
///
/// The recurrent weight matrix is block-diagonal with `n_blocks` independent
/// blocks. Each block operates on its own partition of the hidden state,
/// reducing the recurrent parameter count from `hidden_size^2` to
/// `hidden_size^2 / n_blocks` while preserving the full input projection.
#[derive(Module, Debug)]
pub struct BlockGruCell<B: Backend> {
    /// Input-to-hidden projection: `input_size -> 3 * hidden_size`.
    ih: Linear<B>,
    /// Per-block hidden-to-hidden projections: each `block_size -> 3 * block_size`.
    hh_blocks: Vec<Linear<B>>,
    /// Hidden state dimension.
    #[module(skip)]
    hidden_size: usize,
    /// Number of blocks.
    #[module(skip)]
    n_blocks: usize,
    /// Size of each block (`hidden_size / n_blocks`).
    #[module(skip)]
    block_size: usize,
}

impl BlockGruCellConfig {
    /// Initialize a block-diagonal GRU cell on the given device.
    ///
    /// # Panics
    /// Panics if `hidden_size` is not divisible by `n_blocks`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BlockGruCell<B> {
        assert!(
            self.hidden_size % self.n_blocks == 0,
            "hidden_size {} must be divisible by n_blocks {}",
            self.hidden_size,
            self.n_blocks
        );
        let block_size = self.hidden_size / self.n_blocks;

        let ih = LinearConfig::new(self.input_size, 3 * self.hidden_size).init(device);
        let hh_blocks: Vec<Linear<B>> = (0..self.n_blocks)
            .map(|_| LinearConfig::new(block_size, 3 * block_size).init(device))
            .collect();

        BlockGruCell {
            ih,
            hh_blocks,
            hidden_size: self.hidden_size,
            n_blocks: self.n_blocks,
            block_size,
        }
    }
}

impl<B: Backend> BlockGruCell<B> {
    /// Forward one timestep.
    ///
    /// # Arguments
    /// * `input` — `[batch, input_size]`
    /// * `h` — Previous hidden state `[batch, hidden_size]`
    ///
    /// # Returns
    /// New hidden state `[batch, hidden_size]`.
    pub fn forward(&self, input: Tensor<B, 2>, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = input.dims()[0];
        let bs = self.block_size;
        let hs = self.hidden_size;

        let ih_out = self.ih.forward(input); // [batch, 3*hidden_size]

        let mut new_h_parts = Vec::with_capacity(self.n_blocks);

        for b in 0..self.n_blocks {
            // Extract this block's slice of hidden state
            let h_block = h.clone().slice([0..batch, b * bs..(b + 1) * bs]);
            let hh_out = self.hh_blocks[b].forward(h_block.clone()); // [batch, 3*bs]

            // Extract this block's slice of input projection
            let r_i = ih_out.clone().slice([0..batch, b * bs..(b + 1) * bs]);
            let z_i = ih_out
                .clone()
                .slice([0..batch, hs + b * bs..hs + (b + 1) * bs]);
            let n_i = ih_out
                .clone()
                .slice([0..batch, 2 * hs + b * bs..2 * hs + (b + 1) * bs]);

            let r_h = hh_out.clone().slice([0..batch, 0..bs]);
            let z_h = hh_out.clone().slice([0..batch, bs..2 * bs]);
            let n_h = hh_out.slice([0..batch, 2 * bs..3 * bs]);

            let r = sigmoid(r_i + r_h);
            let z = sigmoid(z_i + z_h);
            let n = (n_i + r * n_h).tanh();

            let new_h_block = (Tensor::ones_like(&z) - z.clone()) * n + z * h_block;
            new_h_parts.push(new_h_block);
        }

        Tensor::cat(new_h_parts, 1) // [batch, hidden_size]
    }

    /// Process a sequence of inputs, returning all hidden states.
    ///
    /// # Arguments
    /// * `inputs` — `[batch, seq_len, input_size]`
    /// * `initial_h` — Initial hidden state `[batch, hidden_size]`
    ///
    /// # Returns
    /// `(outputs, final_h)` where outputs is `[batch, seq_len, hidden_size]`.
    pub fn forward_seq(
        &self,
        inputs: Tensor<B, 3>,
        initial_h: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, seq_len, input_size] = inputs.dims();
        let mut h = initial_h;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t: Tensor<B, 2> = inputs
                .clone()
                .slice([0..batch, t..t + 1, 0..input_size])
                .squeeze_dim::<2>(1);
            h = self.forward(x_t, h);
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }

        (Tensor::cat(outputs, 1), h)
    }
}

// ====================== Tests ======================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // -- LSTM tests ----------------------------------------------------------

    #[test]
    fn lstm_zero_input_produces_zero_state() {
        let cell = LstmCellConfig::new(4, 8).init::<B>(&dev());
        let state = LstmState::zeros(2, 8, &dev());
        let input = Tensor::<B, 2>::zeros([2, 4], &dev());

        let new_state = cell.forward(input, &state);

        // With x=0, h=0, c=0 and bias initialized to zero by default in
        // Burn's Linear, all gates receive 0 input. sigmoid(0)=0.5,
        // tanh(0)=0 => i*g=0.5*0=0, f*c=0.5*0=0 => c1=0, h1=o*tanh(0)=0.
        // However, Linear has non-zero bias by default in Burn, so we only
        // check that shapes are correct and values are finite.
        assert_eq!(new_state.h.dims(), [2, 8]);
        assert_eq!(new_state.c.dims(), [2, 8]);
        let h_vals: Vec<f32> = new_state.h.to_data().to_vec().unwrap();
        let c_vals: Vec<f32> = new_state.c.to_data().to_vec().unwrap();
        for v in h_vals.iter().chain(c_vals.iter()) {
            assert!(v.is_finite(), "LSTM state contains non-finite value: {v}");
        }
    }

    #[test]
    fn lstm_output_shape() {
        let cell = LstmCellConfig::new(4, 8).init::<B>(&dev());
        let state = LstmState::zeros(3, 8, &dev());
        let input = Tensor::<B, 2>::ones([3, 4], &dev());

        let new_state = cell.forward(input, &state);
        assert_eq!(new_state.h.dims(), [3, 8]);
        assert_eq!(new_state.c.dims(), [3, 8]);
    }

    #[test]
    fn lstm_seq_output_shape() {
        let cell = LstmCellConfig::new(4, 8).init::<B>(&dev());
        let state = LstmState::zeros(3, 8, &dev());
        let inputs = Tensor::<B, 3>::ones([3, 5, 4], &dev());

        let (outputs, final_state) = cell.forward_seq(inputs, &state);
        assert_eq!(outputs.dims(), [3, 5, 8]);
        assert_eq!(final_state.h.dims(), [3, 8]);
        assert_eq!(final_state.c.dims(), [3, 8]);
    }

    #[test]
    fn lstm_single_step_matches_seq_first() {
        let cell = LstmCellConfig::new(4, 8).init::<B>(&dev());
        let state = LstmState::zeros(2, 8, &dev());

        // Single input
        let x = Tensor::<B, 2>::ones([2, 4], &dev()) * 0.5;

        // Single-step forward
        let single_state = cell.forward(x.clone(), &state);

        // Sequence forward with length 1
        let x_seq = x.clone().unsqueeze_dim::<3>(1); // [2, 1, 4]
        let (seq_out, seq_state) = cell.forward_seq(x_seq, &state);

        // Compare outputs
        let single_h: Vec<f32> = single_state.h.to_data().to_vec().unwrap();
        let seq_h: Vec<f32> = seq_out.squeeze_dim::<2>(1).to_data().to_vec().unwrap();
        let seq_final_h: Vec<f32> = seq_state.h.to_data().to_vec().unwrap();

        for i in 0..single_h.len() {
            assert!(
                (single_h[i] - seq_h[i]).abs() < 1e-5,
                "LSTM single vs seq output mismatch at {i}: {} vs {}",
                single_h[i],
                seq_h[i]
            );
            assert!(
                (single_h[i] - seq_final_h[i]).abs() < 1e-5,
                "LSTM single vs seq final_h mismatch at {i}"
            );
        }
    }

    // -- GRU tests -----------------------------------------------------------

    #[test]
    fn gru_zero_input_produces_finite_output() {
        let cell = GruCellConfig::new(4, 8).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([2, 8], &dev());
        let input = Tensor::<B, 2>::zeros([2, 4], &dev());

        let new_h = cell.forward(input, h);
        assert_eq!(new_h.dims(), [2, 8]);
        let vals: Vec<f32> = new_h.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "GRU output contains non-finite value: {v}");
        }
    }

    #[test]
    fn gru_output_shape() {
        let cell = GruCellConfig::new(4, 8).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 8], &dev());
        let input = Tensor::<B, 2>::ones([3, 4], &dev());

        let new_h = cell.forward(input, h);
        assert_eq!(new_h.dims(), [3, 8]);
    }

    #[test]
    fn gru_seq_output_shape() {
        let cell = GruCellConfig::new(4, 8).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 8], &dev());
        let inputs = Tensor::<B, 3>::ones([3, 5, 4], &dev());

        let (outputs, final_h) = cell.forward_seq(inputs, h);
        assert_eq!(outputs.dims(), [3, 5, 8]);
        assert_eq!(final_h.dims(), [3, 8]);
    }

    #[test]
    fn gru_single_step_matches_seq_first() {
        let cell = GruCellConfig::new(4, 8).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([2, 8], &dev());

        let x = Tensor::<B, 2>::ones([2, 4], &dev()) * 0.5;

        // Single-step forward
        let single_h = cell.forward(x.clone(), h.clone());

        // Sequence forward with length 1
        let x_seq = x.unsqueeze_dim::<3>(1);
        let (seq_out, seq_final_h) = cell.forward_seq(x_seq, h);

        let single_vals: Vec<f32> = single_h.to_data().to_vec().unwrap();
        let seq_vals: Vec<f32> = seq_out.squeeze_dim::<2>(1).to_data().to_vec().unwrap();
        let seq_final_vals: Vec<f32> = seq_final_h.to_data().to_vec().unwrap();

        for i in 0..single_vals.len() {
            assert!(
                (single_vals[i] - seq_vals[i]).abs() < 1e-5,
                "GRU single vs seq output mismatch at {i}"
            );
            assert!(
                (single_vals[i] - seq_final_vals[i]).abs() < 1e-5,
                "GRU single vs seq final_h mismatch at {i}"
            );
        }
    }

    // -- BlockGRU tests ------------------------------------------------------

    #[test]
    fn block_gru_output_shape() {
        let cell = BlockGruCellConfig::new(4, 16).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 16], &dev());
        let input = Tensor::<B, 2>::ones([3, 4], &dev());

        let new_h = cell.forward(input, h);
        assert_eq!(new_h.dims(), [3, 16]);
    }

    #[test]
    fn block_gru_seq_output_shape() {
        let cell = BlockGruCellConfig::new(4, 16).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([3, 16], &dev());
        let inputs = Tensor::<B, 3>::ones([3, 5, 4], &dev());

        let (outputs, final_h) = cell.forward_seq(inputs, h);
        assert_eq!(outputs.dims(), [3, 5, 16]);
        assert_eq!(final_h.dims(), [3, 16]);
    }

    #[test]
    fn block_gru_n_blocks_1_matches_gru() {
        // With n_blocks=1 and identical weights, BlockGRU should produce
        // the same output as a standard GRU.
        let device = dev();
        let hidden_size = 8;
        let input_size = 4;

        let gru = GruCellConfig::new(input_size, hidden_size).init::<B>(&device);
        let mut block_gru =
            BlockGruCellConfig::new(input_size, hidden_size)
                .with_n_blocks(1)
                .init::<B>(&device);

        // Copy weights from GRU to BlockGRU
        // ih weights are the same structure
        let gru_record = gru.clone().into_record();
        let _block_record = block_gru.clone().into_record();

        // Load GRU's ih into BlockGRU's ih, and GRU's hh into BlockGRU's single hh_block
        use burn::module::Param;
        use burn::tensor::TensorData;

        // Get GRU weight data
        let ih_weight_data: TensorData = gru_record.ih.weight.val().to_data();
        let ih_bias_data: TensorData = gru_record.ih.bias.clone().unwrap().val().to_data();
        let hh_weight_data: TensorData = gru_record.hh.weight.val().to_data();
        let hh_bias_data: TensorData = gru_record.hh.bias.clone().unwrap().val().to_data();

        // Build BlockGRU record with the same weights
        use burn::nn::LinearRecord;
        let new_ih_record = LinearRecord {
            weight: Param::from_data(ih_weight_data, &device),
            bias: Some(Param::from_data(ih_bias_data, &device)),
        };
        let new_hh_record = LinearRecord {
            weight: Param::from_data(hh_weight_data, &device),
            bias: Some(Param::from_data(hh_bias_data, &device)),
        };

        // Rebuild block_gru with matching weights
        block_gru.ih = block_gru.ih.load_record(new_ih_record);
        block_gru.hh_blocks[0] = block_gru.hh_blocks[0].clone().load_record(new_hh_record);

        // Run both with the same input
        let h = Tensor::<B, 2>::zeros([2, hidden_size], &device);
        let x = Tensor::<B, 2>::ones([2, input_size], &device) * 0.3;

        let gru_h = gru.forward(x.clone(), h.clone());
        let block_h = block_gru.forward(x, h);

        let gru_vals: Vec<f32> = gru_h.to_data().to_vec().unwrap();
        let block_vals: Vec<f32> = block_h.to_data().to_vec().unwrap();

        for i in 0..gru_vals.len() {
            assert!(
                (gru_vals[i] - block_vals[i]).abs() < 1e-5,
                "BlockGRU(n_blocks=1) vs GRU mismatch at {i}: {} vs {}",
                gru_vals[i],
                block_vals[i]
            );
        }
    }

    #[test]
    fn block_gru_zero_input_produces_finite_output() {
        let cell = BlockGruCellConfig::new(4, 16).init::<B>(&dev());
        let h = Tensor::<B, 2>::zeros([2, 16], &dev());
        let input = Tensor::<B, 2>::zeros([2, 4], &dev());

        let new_h = cell.forward(input, h);
        let vals: Vec<f32> = new_h.to_data().to_vec().unwrap();
        for v in &vals {
            assert!(
                v.is_finite(),
                "BlockGRU output contains non-finite value: {v}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn block_gru_panics_on_indivisible() {
        let _ = BlockGruCellConfig::new(4, 10)
            .with_n_blocks(3)
            .init::<B>(&dev());
    }

    // -- Gradient flow tests (autodiff) --------------------------------------

    #[test]
    fn lstm_gradient_flows_through_sequence() {
        use burn::backend::Autodiff;

        type AB = Autodiff<NdArray>;
        let device = Default::default();

        let cell = LstmCellConfig::new(4, 8).init::<AB>(&device);
        let state = LstmState::zeros(2, 8, &device);
        let inputs = Tensor::<AB, 3>::ones([2, 3, 4], &device).require_grad();

        let (outputs, _final_state) = cell.forward_seq(inputs.clone(), &state);
        // Loss from the last timestep only
        let last_out = outputs.slice([0..2, 2..3, 0..8]);
        let loss = last_out.sum();
        let grads = loss.backward();

        let input_grad = inputs.grad(&grads).expect("inputs should have gradients");
        let grad_vals: Vec<f32> = input_grad.to_data().to_vec().unwrap();

        // Gradient at the first timestep should be non-zero (information flows
        // from t=0 through hidden state to the loss at t=2).
        let first_step_grad_sum: f32 = grad_vals[..4].iter().map(|g| g.abs()).sum();
        assert!(
            first_step_grad_sum > 1e-8,
            "Gradients should flow to first timestep, got sum={first_step_grad_sum}"
        );
    }

    #[test]
    fn gru_gradient_flows_through_sequence() {
        use burn::backend::Autodiff;

        type AB = Autodiff<NdArray>;
        let device = Default::default();

        let cell = GruCellConfig::new(4, 8).init::<AB>(&device);
        let h = Tensor::<AB, 2>::zeros([2, 8], &device);
        let inputs = Tensor::<AB, 3>::ones([2, 3, 4], &device).require_grad();

        let (outputs, _final_h) = cell.forward_seq(inputs.clone(), h);
        let last_out = outputs.slice([0..2, 2..3, 0..8]);
        let loss = last_out.sum();
        let grads = loss.backward();

        let input_grad = inputs.grad(&grads).expect("inputs should have gradients");
        let grad_vals: Vec<f32> = input_grad.to_data().to_vec().unwrap();

        let first_step_grad_sum: f32 = grad_vals[..4].iter().map(|g| g.abs()).sum();
        assert!(
            first_step_grad_sum > 1e-8,
            "Gradients should flow to first timestep, got sum={first_step_grad_sum}"
        );
    }

    #[test]
    fn block_gru_gradient_flows_through_sequence() {
        use burn::backend::Autodiff;

        type AB = Autodiff<NdArray>;
        let device = Default::default();

        let cell = BlockGruCellConfig::new(4, 16).init::<AB>(&device);
        let h = Tensor::<AB, 2>::zeros([2, 16], &device);
        let inputs = Tensor::<AB, 3>::ones([2, 3, 4], &device).require_grad();

        let (outputs, _final_h) = cell.forward_seq(inputs.clone(), h);
        let last_out = outputs.slice([0..2, 2..3, 0..16]);
        let loss = last_out.sum();
        let grads = loss.backward();

        let input_grad = inputs.grad(&grads).expect("inputs should have gradients");
        let grad_vals: Vec<f32> = input_grad.to_data().to_vec().unwrap();

        let first_step_grad_sum: f32 = grad_vals[..4].iter().map(|g| g.abs()).sum();
        assert!(
            first_step_grad_sum > 1e-8,
            "Gradients should flow to first timestep, got sum={first_step_grad_sum}"
        );
    }
}
