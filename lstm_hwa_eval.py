import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
    WeightNoiseType,
)



def gen_rpu_config():
    rpu_config = InferenceRPUConfig()

    # modifier for hwa training
    rpu_config.modifier.type = WeightModifierType.PCM_NOISE
    rpu_config.modifier.std_dev = 5.0
    rpu_config.modifier.pdrop = 0.01

    
    # forward
    rpu_config.forward.out_res = 8
    rpu_config.forward.inp_res = 8
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.out_bound = 10.0
    rpu_config.forward.ir_drop_g_ratio = 571428.57
    rpu_config.forward.ir_drop = 1.0
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.w_noise_type = WeightNoiseType.PCM_READ
    rpu_config.forward.w_noise = 0.0175
    rpu_config.forward.inp_bound = 1.0

    
    # mapping
    rpu_config.mapping.max_input_size = 512
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True

    # learn input range
    rpu_config.pre_post.input_range.enable = True
    

    # noise model
    rpu_config.noise_model = PCMLikeNoiseModel()
    rpu_config.noise_model.g_max = 25.0

    rpu_config.drift_compensation = GlobalDriftCompensation()
    