import dataclasses



@dataclasses.dataclass
class LSTM_FP_Config:
    # model parameters
    embedding_dim: int = 650
    hidden_size: int = 650
    num_layers: int = 2
    dropout: float = 0.5
    batch_size: int = 20
    seq_length: int = 35
    epochs: int = 60


    # fp training parameters
    lr: float = 20.0
    max_grad_norm: float = 0.25
    epochs: int = 60
    lr_decay_factor: float = 5 #lr/lr_decay_factor if valid loss not improved
    weight_decay: float = 1e-5

    # fp evaluation results
    error: float = 0.729 # best error rate on test set
    ppl: float = 80.856 # best perplexity on test set


    