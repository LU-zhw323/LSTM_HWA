import dataclasses



@dataclasses.dataclass
class LSTM_HWA_Config:
    # model parameters
    embedding_dim: int = 650
    hidden_size: int = 650
    num_layers: int = 2
    dropout: float = 0.5
    batch_size: int = 20
    seq_length: int = 35


    # fp training parameters
    fp_lr: float = 20.0
    fp_max_grad_norm: float = 0.25
    fp_epochs: int = 40
    fp_lr_decay_factor: float = 5 #lr/lr_decay_factor if valid loss not improved
    fp_error: float = 0.726 # best error rate on test set

    