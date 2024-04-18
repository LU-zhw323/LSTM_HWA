import model
import json
from math import log10
import math
import os
import argparse
import time
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, WeightClipType, WeightModifierType, WeightNoiseType, WeightRemapType
from numpy.core.function_base import logspace
import torch
from typing import Tuple
from torch import tensor, device, FloatTensor, Tensor, transpose, save, load
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.functional import one_hot

import numpy as np
import h5py
from aihwkit.nn import AnalogSequential, AnalogRNN, AnalogLinear, AnalogLSTMCellCombinedWeight
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    UnitCellRPUConfig,
    SingleRPUConfig,
    BufferedTransferCompound,
    SoftBoundsDevice,
    ConstantStepDevice,
    MappingParameter,
    IOParameters,
    UpdateParameters,
)
from aihwkit.simulator.rpu_base import cuda

import data
import csv


def inference(analog_model, evaluate, test_data, args, file_name, group_name, com):
    print('=' * 89)
    print("Inference")
    print('-' * 89)
    start_time = 60
    max_inference_time = 31536000
    n_times = 9
    t_inference_list = [
            0.0] + logspace(0, log10(float(max_inference_time)), n_times).tolist()
    dtype = np.dtype([
        ('noise', np.float32),
        ('time', np.float32), 
        ('loss', np.float32), 
        ('ppl', np.float32)
    ])
    inference_data = np.empty(len(t_inference_list), dtype=dtype)
    try:
        analog_model.eval()
        with h5py.File(file_name, 'a', driver='mpio', comm=com) as f:
            task_group = f.require_group(group_name)
            #t_inference in second
            for i, t_inference in enumerate(t_inference_list):
                analog_model.drift_analog_weights(t_inference)
                inference_loss = evaluate(test_data)
                print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
                t_inference,inference_loss, math.exp(inference_loss)))
                inference_data[i] = (args.noise, t_inference, inference_loss, math.exp(inference_loss))
                
            if 'inference_results' in task_group:
                del task_group['inference_results']
            task_group.create_dataset('inference_results', data=inference_data)
            print('=' * 89)
            print()
        com.Barrier()
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from Inference early')


def inference_noise_model(analog_model, evaluate, test_data, args, file_name, group_name, com):
    print('=' * 89)
    print("Inference")
    print('-' * 89)
    start_time = 60
    max_inference_time = 31536000
    n_times = 9
    t_inference_list = [
            0.0] + logspace(0, log10(float(max_inference_time)), n_times).tolist()
    dtype = np.dtype([
        ('program_noise', np.float32),
        ('read_noise', np.float32), 
        ('drift', np.float32), 
        ('gmin', np.float32), 
        ('gmax', np.float32), 
        ('time', np.float32),
        ('loss', np.float32), 
        ('ppl', np.float32)
    ])
    inference_data = np.empty(len(t_inference_list), dtype=dtype)
    try:
        analog_model.eval()
        with h5py.File(file_name, 'a', driver='mpio', comm=com) as f:
            task_group = f.require_group(group_name)
            #t_inference in second
            for i, t_inference in enumerate(t_inference_list):
                analog_model.drift_analog_weights(t_inference)
                inference_loss = evaluate(test_data)
                print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
                t_inference,inference_loss, math.exp(inference_loss)))
                inference_data[i] = (
                    args.inference_progm_noise, 
                    args.inference_read_noise, 
                    args.drift,
                    args.gmin,
                    args.gmax, 
                    t_inference, 
                    inference_loss, 
                    math.exp(inference_loss))
                
            if 'inference_results' in task_group:
                del task_group['inference_results']
            task_group.create_dataset('inference_results', data=inference_data)
            print('=' * 89)
            print()
        com.Barrier()
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from Inference early')
