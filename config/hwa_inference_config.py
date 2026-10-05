import dataclasses
from typing import List



@dataclasses.dataclass
class LSTM_HWA_INFERENCE_Config:
    """Model shape, FP baseline and sweep grid of `hwa_inference.py`."""

    # model parameters
    embedding_dim: int = 650
    """Word embedding size. Must match the checkpoints."""
    hidden_size: int = 650
    """LSTM hidden state size per layer. Must match the checkpoints."""
    num_layers: int = 2
    """Number of stacked LSTM layers. Must match the checkpoints."""
    dropout: float = 0.5
    """Dropout probability. Inactive in evaluation."""
    batch_size: int = 20
    """Number of parallel token streams in the test batcher."""
    seq_length: int = 35
    """Tokens per test batch step."""

    # fp baseline
    fp_error: float = 0.72794
    """Test error rate of the FP model, fraction. The 100% point of the normalized accuracy."""

    # hwa training parameters
    hwa_noise_scale: float = 5.0
    """Std of the PCM weight-noise modifier in the RPU config. Inactive in evaluation."""
    pdrop: float = 0.01
    """Weight drop probability of the modifier in the RPU config. Inactive in evaluation."""

    # noise model parameters
    noise_scale: List[float] = dataclasses.field(default_factory=lambda: [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    """Multipliers of the PCM programming and read noise of the aihwkit PCM model. Unitless."""
    drift_scale: List[float] = dataclasses.field(default_factory=lambda: [0.05, 0.5, 1.0])
    """Multipliers of the PCM drift coefficient of the aihwkit PCM model. Unitless."""
    g_min: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.005, 0.05, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    """Minimum device conductances, uS. The memory window is g_max - g_min."""
    g_max: float = 25.0
    """Maximum device conductance, uS."""

    # hwa evaluation parameters
    num_evals: int = 25
    """Evaluations per configuration, averaged. All share one programming-noise draw; each draws new read noise."""
    inference_time: List[float] = dataclasses.field(default_factory=lambda: [1, 3600, 3600*24, 3600*24*7, 3600*24*365])
    """Times after programming at which the weights are drifted, s."""
