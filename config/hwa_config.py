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
    epochs: int = 60



    # hwa training parameters
    hwa_noise_scale: float = 5.0
    pdrop: float = 0.01
    lr: float = 0.01
    lr_decay_factor: float = 0.9 # applied after each epoch if valid loss not improved
    momentum: float = 0.9
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-5

    # noise model parameters
    noise_scale: float = 1.0
    drift_scale: float = 1.0
    g_min: float = 0.0
    g_max: float = 25.0

    # hwa evaluation parameters
    num_evals: int = 1
    t_inference: float = 365 * 24 * 60 * 60 # 1 year


    