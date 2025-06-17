import dataclasses
from typing import List

@dataclasses.dataclass
class DeviceConfig:
    name: str
    drift_scale: float
    g_min: float
    g_max: float

@dataclasses.dataclass
class LSTM_HWA_INFERENCE_DEVICE_Config:
    # model parameters
    embedding_dim: int = 650
    hidden_size: int = 650
    num_layers: int = 2
    dropout: float = 0.5
    batch_size: int = 20
    seq_length: int = 35
    epochs: int = 60

    # fp baseline
    fp_error: float = 0.72794

    # hwa training parameters
    hwa_noise_scale: float = 5.0
    pdrop: float = 0.01
    lr: float = 0.01
    lr_decay_factor: float = 0.9 # applied after each epoch if valid loss not improved
    momentum: float = 0.9
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-5

    # noise model parameters
    base_drift_coeff: float = 0.049
    noise_scale: List[float] = dataclasses.field(default_factory=lambda: [2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0])
    devices: List[DeviceConfig] = dataclasses.field(default_factory=lambda: [
        DeviceConfig(name='liner_1', drift_scale=0.01/0.049, g_min=0.5, g_max=20.0),
        DeviceConfig(name='liner_2', drift_scale=0.02/0.049, g_min=1.0, g_max=300.0),
        DeviceConfig(name='homo', drift_scale=0.025/0.049, g_min=1.0, g_max=20.0),
        DeviceConfig(name='opt', drift_scale=0.04/0.049, g_min=0.0, g_max=75.0),
        DeviceConfig(name='cpl', drift_scale=0.03/0.049, g_min=0.1, g_max=100.0),
    ])
    # hwa evaluation parameters
    num_evals: int = 25
    inference_time: List[float] = dataclasses.field(default_factory=lambda: [1, 3600, 3600*24, 3600*24*7, 3600*24*365])


    