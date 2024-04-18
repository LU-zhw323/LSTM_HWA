import utils
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
from mpi4py import MPI
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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
def parse_args():
    parser = argparse.ArgumentParser(description='Model training script.')
    parser.add_argument('--task_id', type=int, help='Task ID from SLURM array job')
    args = parser.parse_args()
    task_id = args.task_id
    with open('parameters.json', 'r') as f:
        params = json.load(f)
    param = params[str(task_id)]

    return param

class Params:
    def __init__(self):
        pass

def set_param():
    param = parse_args()
    print()
    print('=' * 89)
    print("Parameters")
    print('-' * 89)
    #If you assigned parameters in the parameters.json file, please replace below parameters
    #For instance, if your parameter.json has task of:
    # "1": {"lr": 0.01, "dropout": 0.5, "epoch": 60}
    # Then you need to replace args.lr as args.lr = params['lr']
    args = Params()

    args.noise = 3.4
    print(f"HWA Training Noise: {args.noise}")

    args.w_drop = 0.01
    print(f"Weight Drop: {args.w_drop}")

    args.drift = 1
    print(f"Drift: {args.drift}")

    args.inference_noise = 1
    print(f"Inference Noise: {args.inference_noise}")

    args.mwindow = 1
    print(f"Memory Window: {args.mwindow}")


    args.model = 'LSTM'
    args.data = './data/ptb'
    args.emsize = 650
    args.nhid = 650
    args.nlayers = 2
    args.batch_size = 20
    args.seq_len = 35
    args.tied = False
    args.log_interval = 200
    args.save = 'lstm_hwa.pt'
    print('=' * 89)
    print()
    return args


###############################################################################
# Set Parameter
###############################################################################
args = set_param()


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(DEVICE)

corpus = data.Corpus(args.data)
eval_batch_size = 20
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
ntokens = len(corpus.dictionary)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def gen_rpu_config():
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.type = WeightModifierType.PCM_NOISE
    rpu_config.modifier.pcm_t0 = 20
    rpu_config.modifier.pdrop = args.w_drop
    rpu_config.modifier.std_dev = args.noise

    rpu_config.forward = IOParameters()
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


###############################################################################
# Build the model
###############################################################################
model_save_path = './model/lstm_hwa.th'
hwa_model = model.RNNModel(args.model, ntokens, 650, 650, 2, 0.5, False).to(DEVICE)
analog_model = convert_to_analog(hwa_model,gen_rpu_config())
analog_model.load_state_dict(
        torch.load(model_save_path, map_location=DEVICE)
    )
analog_model.rnn.flatten_parameters()
#Since in the forward, it will perform log_softmax() on the output 
#Therefore, using NLLLoss here is equivalent to CrossEntropyLoss
criterion = nn.NLLLoss()

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    analog_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = analog_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = analog_model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = analog_model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

print()
comm.Barrier()
utils.inference(analog_model, evaluate, test_data, args, './result/lstm_inf.h5', f"task_noise_{args.noise}", comm)
"""print('=' * 89)
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
    with h5py.File('./result/best_accuracy.h5', 'a') as f:
        task_group = f.require_group(f"task_noise_{args.noise}")
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
except KeyboardInterrupt:
    print('=' * 89)
    print('Exiting from Inference early')
"""