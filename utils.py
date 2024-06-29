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
import portalocker


def inference(analog_model, evaluate, test_data, args, file_name, group_name):
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
        #t_inference in second
        for i, t_inference in enumerate(t_inference_list):
            analog_model.drift_analog_weights(t_inference)
            inference_loss = evaluate(test_data)
            print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
            t_inference,inference_loss, math.exp(inference_loss)))
            inference_data[i] = (args.noise, t_inference, inference_loss, math.exp(inference_loss))
    except KeyboardInterrupt:
        print('=' * 89)
        print('Exiting from Inference early')
    attempt = 0
    release = 10
    while attempt < release:
        try:
            with h5py.File(file_name, 'a') as f:
                task_group = None
                if not group_name in f:
                    task_group = f.create_group(group_name)
                else:
                    task_group = f[group_name]
                print(task_group)
                if 'inference_results' in task_group:
                    del task_group['inference_results']
                task_group.create_dataset('inference_results', data=inference_data)

                group_names = list(f.keys())
                print(f"Groups in '{file_name}': {group_names}")
                print('=' * 89)
                print()
                break
        except OSError as e:
            attempt += 1
            if attempt < release:
                print(f"Attempt {attempt}: File is locked, retrying in {10} seconds...")
                time.sleep(10)
                continue
            else:
                print('=' * 89)
                print(f"Exceed Maximum Attempt at {attempt} attempts: {e}")
                break



def inference_noise_model(analog_model, evaluate, test_data, args, file_name, group_name, model_type, encoder):
    print('=' * 89)
    print("Inference")
    print(f'File: {file_name}, Group: {group_name}, Data: {args.task_param}')
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
    analog_model.eval()
    #t_inference in second
    try:
        for i, t_inference in enumerate(t_inference_list):
                    analog_model.drift_analog_weights(t_inference)
                    inference_loss = evaluate(test_data,encoder, model_type)
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
    except KeyboardInterrupt:
            print('=' * 89)
            print('Exiting from Inference early')
            return

    attempt = 0
    release = 20
    while attempt < release:
        try:
            with h5py.File(file_name, 'a') as f:
                task_group = None
                if not group_name in f:
                    task_group = f.create_group(group_name)
                else:
                    task_group = f[group_name]
                print(task_group)
                if str(args.task_param) in task_group:
                    del task_group[str(args.task_param)]
                task_group.create_dataset(str(args.task_param), data=inference_data)
                print('=' * 89)
                print()
                break
        except OSError as e:
            attempt += 1
            if attempt < release:
                print(f"Attempt {attempt}: File is locked, retrying in {10} seconds...")
                time.sleep(20)
                continue
            else:
                print('=' * 89)
                print(f"Exceed Maximum Attempt at {attempt} attempts: {e}")
                break


def inference_time(analog_model, evaluate, test_data, args, file_name, group_name, model_type, encoder, t_inference):
    print('=' * 89)
    print("Inference")
    print(f'File: {file_name}, Group: {group_name}, Data: {args.task_param}')
    print('-' * 89)
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
    inference_data = np.empty(1, dtype=dtype)
    analog_model.eval()
    #t_inference in second
    try:
        analog_model.drift_analog_weights(t_inference)
        inference_loss = evaluate(test_data,encoder, model_type)
        print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
        t_inference,inference_loss, math.exp(inference_loss)))
        inference_data[0] = (
            args.inference_progm_noise, 
            args.inference_read_noise, 
            args.drift,
            args.gmin,
            args.gmax, 
            t_inference, 
            inference_loss, 
            math.exp(inference_loss))
    except KeyboardInterrupt:
            print('=' * 89)
            print('Exiting from Inference early')
            return

    attempt = 0
    release = 20
    while attempt < release:
        try:
            with h5py.File(file_name, 'a') as f:
                task_group = None
                if not group_name in f:
                    task_group = f.create_group(group_name)
                else:
                    task_group = f[group_name]
                print(task_group)
                if str(args.task_param) in task_group:
                    del task_group[str(args.task_param)]
                task_group.create_dataset(str(args.task_param), data=inference_data)
                print('=' * 89)
                print()
                break
        except OSError as e:
            attempt += 1
            if attempt < release:
                print(f"Attempt {attempt}: File is locked, retrying in {10} seconds...")
                time.sleep(20)
                continue
            else:
                print('=' * 89)
                print(f"Exceed Maximum Attempt at {attempt} attempts: {e}")
                break


def inference_time_avg(analog_model, evaluate, test_data, args, file_name, group_name, model_type, encoder, t_inference):
    print('=' * 89)
    print("Inference")
    print(f'File: {file_name}, Group: {group_name}, Data: {args.task_param}')
    print('-' * 89)
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
    inference_data = np.empty(1, dtype=dtype)
    analog_model.eval()
    #t_inference in second
    try:
        total_inference_loss = 0.0
        num_iterations = 25
        for i in range(num_iterations):
            analog_model.drift_analog_weights(t_inference)
            inference_loss = evaluate(test_data,encoder, model_type)
            print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
            t_inference,inference_loss, math.exp(inference_loss)))
            total_inference_loss += inference_loss

        average_inference_loss = total_inference_loss / num_iterations
        average_test_ppl = math.exp(average_inference_loss)

        inference_data[0] = (
            args.inference_progm_noise, 
            args.inference_read_noise, 
            args.drift,
            args.gmin,
            args.gmax, 
            t_inference, 
            average_inference_loss, 
            average_test_ppl)
    except KeyboardInterrupt:
            print('=' * 89)
            print('Exiting from Inference early')
            return

    attempt = 0
    release = 20
    while attempt < release:
        try:
            with h5py.File(file_name, 'a') as f:
                task_group = None
                if not group_name in f:
                    task_group = f.create_group(group_name)
                else:
                    task_group = f[group_name]
                if str(args.task_param) in task_group:
                    del task_group[str(args.task_param)]
                task_group.create_dataset(str(args.task_param), data=inference_data)
                print(task_group)
                print('=' * 89)
                print()
                break
        except OSError as e:
            attempt += 1
            if attempt < release:
                print(f"Attempt {attempt}: File is locked, retrying in {10} seconds...")
                time.sleep(20)
                continue
            else:
                print('=' * 89)
                print(f"Exceed Maximum Attempt at {attempt} attempts: {e}")
                break



def inference_base(analog_model, evaluate, test_data, args, file_name, group_name, model_type, encoder, t_inference):
    print('=' * 89)
    print("Inference")
    print(f'File: {file_name}, Group: {group_name}, Data: {args.task_param}')
    print('-' * 89)
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
    inference_data = np.empty(1, dtype=dtype)
    analog_model.eval()
    #t_inference in second
    try:
        total_inference_loss = 0.0
        num_iterations = 25
        for i in range(num_iterations):
            analog_model.drift_analog_weights(t_inference)
            inference_loss = evaluate(test_data,encoder, model_type)
            print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
            t_inference,inference_loss, math.exp(inference_loss)))
            total_inference_loss += inference_loss

        average_inference_loss = total_inference_loss / num_iterations
        average_test_ppl = math.exp(average_inference_loss)

        inference_data[0] = (
            args.inference_progm_noise, 
            args.inference_read_noise, 
            args.drift,
            args.gmin,
            args.gmax, 
            t_inference, 
            average_inference_loss, 
            average_test_ppl)
    except KeyboardInterrupt:
            print('=' * 89)
            print('Exiting from Inference early')
            return

    attempt = 0
    release = 20
    while attempt < release:
        try:
            with h5py.File(file_name, 'a') as f:
                task_group = None
                if not group_name in f:
                    task_group = f.create_group(group_name)
                else:
                    task_group = f[group_name]
                if str(args.task_param) in task_group:
                    del task_group[str(args.task_param)]
                task_group.create_dataset(str(args.task_param), data=inference_data)
                print(task_group)
                print('=' * 89)
                print()
                break
        except OSError as e:
            attempt += 1
            if attempt < release:
                print(f"Attempt {attempt}: File is locked, retrying in {10} seconds...")
                time.sleep(20)
                continue
            else:
                print('=' * 89)
                print(f"Exceed Maximum Attempt at {attempt} attempts: {e}")
                break
