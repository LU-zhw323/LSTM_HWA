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
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
import torch
from typing import Tuple
from torch import tensor, device, FloatTensor, Tensor, transpose, save, load
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.functional import one_hot
import types
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
class Params:
    def __init__(self):
        pass
    

def set_param():
    print()
    print('=' * 89)
    print("Parameters")
    print('-' * 89)
    #If you assigned parameters in the parameters.json file, please replace below parameters
    #For instance, if your parameter.json has task of:
    # "1": {"lr": 0.01, "dropout": 0.5, "epoch": 60}
    # Then you need to replace args.lr as args.lr = params['lr']
    args = Params()
    args.lr = 0.01
    print(f"Learning Rate: {args.lr}")

    args.dropout = 0.5
    print(f"Dropout Rate: {args.dropout}")

    args.noise = 3.4
    print(f"Noise: {args.noise}")

    args.clip = 10.0
    print(f"Gradient Clipping: {args.clip}")

    args.w_drop = 0.01
    print(f"Weight Drop: {args.w_drop}")

    args.lr_decay = 0.9
    print(f"Learning Rate Decay: {args.lr_decay}")

    args.w_decay = 1e-5
    print(f"Weight Decay: {args.w_decay}")

    args.mom = 0.9
    print(f"Momentum: {args.mom}")

    args.epochs = 60
    print(f"Epochs: {args.epochs}")

    args.drift = 0.0
    

    args.inference_progm_noise = 0.0



    # Long term Read fluctuations (short term read noise in IO Parameter)
    args.inference_read_noise = 0.0


    # Default = 0
    args.gmin = 0.0


    # Default = 25
    args.gmax = 25.0

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

def gen_rpu_config(args):
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.type = WeightModifierType.PCM_NOISE
    rpu_config.modifier.pcm_t0 = 20.0
    rpu_config.modifier.pdrop = args.w_drop
    rpu_config.modifier.std_dev = args.noise
    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale = 0.0,
        read_noise_scale = 0.0,
        drift_scale = 0.0,
        g_converter=SinglePairConductanceConverter(g_min=0, g_max=25),
        )
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


###############################################################################
# Build the model
###############################################################################
def new_forward(self, encoded_input, hidden):
    emb = self.drop(encoded_input)
    output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)
    decoded = self.decoder(output)
    decoded = decoded.view(-1, self.ntoken)
    return torch.nn.functional.log_softmax(decoded, dim=1), hidden

analog_model = None
group_name = None
encoder = None
model_type = 'HWA'
if(model_type == 'FP'):
    model_save_path = './model/lstm_fp.pt'
    pre_model = torch.load(model_save_path).to(DEVICE)
    pre_model.rnn.flatten_parameters()
    del pre_model.encoder
    pre_model.forward = types.MethodType(new_forward, pre_model)
    analog_model = convert_to_analog(pre_model, gen_rpu_config())
    analog_model.rnn.flatten_parameters()
    encoder =torch.load('./model/encoder.pt').to(DEVICE)
    group_name = f'FP'
elif(model_type == 'HWA'):
    model_save_path = './model/lstm_fp.pt'
    pre_model = torch.load(model_save_path)
    pre_model.rnn.flatten_parameters()
    del pre_model.encoder
    pre_model.forward = types.MethodType(new_forward, pre_model)

    analog_model = convert_to_analog(pre_model, gen_rpu_config(args))
    analog_model.load_state_dict(
            torch.load('./model/lstm_hwa_3.4.th', map_location = DEVICE),
            load_rpu_config=True
        )
    analog_model.rnn.flatten_parameters()
    encoder =torch.load('./model/encoder.pt').to(DEVICE)
    group_name = f'HWA'
else:
    print(f'No such Model: {model_type}')
#Since in the forward, it will perform log_softmax() on the output 
#Therefore, using NLLLoss here is equivalent to CrossEntropyLoss
criterion = nn.NLLLoss()

def evaluate(data_source, encoder, model_type):
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
                data = encoder(data)
                output, hidden = analog_model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

print()

###############################################################################
# Specify time
###############################################################################
h5_file = f'./result/lstm_inf_baseline.h5'
#day
#time = 86400.0
#week
time = 0.0
#month
#time = 2678400.0
#three month
#time = time * 3.0
#year
#time = time * 4.0

args.task_param = f"gmax{args.gmax}_gmin{args.gmin}_n{args.inference_progm_noise}_d{args.drift}"
utils.inference_base(analog_model, evaluate, test_data, args, h5_file, group_name, model_type, encoder, time)
