# LSTM, GRU, and Block GRU

Recurrent cells for temporal reasoning under partial observability. Every game AI paper (AlphaStar, SCC, JueWu) uses LSTM or GRU for sequence processing.

## LSTM Cell

```rust,ignore
use rl4burn::{LstmCell, LstmCellConfig, LstmState};

let cell = LstmCellConfig::new(input_size, hidden_size).init(&device);
let state = LstmState::zeros(batch_size, hidden_size, &device);

// Single step
let new_state = cell.forward(input, &state);

// Full sequence
let (outputs, final_state) = cell.forward_seq(inputs, &state);
// outputs: [batch, seq_len, hidden_size]
```

## GRU Cell

Same API, simpler internals (2 gates vs LSTM's 3). Uses PyTorch's convention for reset gate application.

```rust,ignore
use rl4burn::{GruCell, GruCellConfig};

let cell = GruCellConfig::new(input_size, hidden_size).init(&device);
let h = Tensor::zeros([batch_size, hidden_size], &device);
let new_h = cell.forward(input, h);
```

## Block GRU (DreamerV3)

Block-diagonal GRU reduces recurrent parameters by a factor of `n_blocks`. The recurrent weight matrix is split into independent blocks, each operating on a partition of the hidden state.

DreamerV3 uses 8 blocks with a 4096-dim hidden state, reducing parameters from 16M to 2M.

```rust,ignore
use rl4burn::{BlockGruCell, BlockGruCellConfig};

let cell = BlockGruCellConfig::new(input_size, hidden_size)
    .with_n_blocks(8)
    .init(&device);
```

When `n_blocks = 1`, Block GRU is identical to standard GRU.
