import dataclasses



@dataclasses.dataclass
class LSTM_FP_Config:
    """Model shape and training hyperparameters of the FP LSTM language model on PTB."""

    # model parameters
    embedding_dim: int = 650
    """Word embedding size."""
    hidden_size: int = 650
    """LSTM hidden state size, per layer."""
    num_layers: int = 2
    """Number of stacked LSTM layers."""
    dropout: float = 0.5
    """Dropout probability between LSTM layers and before the output layer."""
    batch_size: int = 20
    """Number of parallel token streams per batch."""
    seq_length: int = 35
    """Truncated backpropagation length, tokens."""
    epochs: int = 60
    """Number of training epochs."""

    # fp training parameters
    lr: float = 20.0
    """Initial SGD learning rate."""
    max_grad_norm: float = 0.25
    """Gradient clipping threshold on the L2 norm over all parameters."""
    lr_decay_factor: float = 5
    """Divisor applied to the learning rate after an epoch without validation-loss improvement."""
    weight_decay: float = 1e-5
    """SGD weight decay."""
