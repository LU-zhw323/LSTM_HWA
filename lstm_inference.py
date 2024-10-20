import types
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
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


    args.noise = 3.4

    args.w_drop = 0.01
    

    args.drift = 1.0
    

    args.inference_progm_noise = 1.0



    # Long term Read fluctuations (short term read noise in IO Parameter)
    args.inference_read_noise = 1.0


    # Default = 0
    args.gmin = 0


    # Default = 25
    args.gmax = 25
    print(f"Drift: {args.drift}")
    print(f"Inference Program Noise: {args.inference_progm_noise}")
    print(f"Inference Read Noise: {args.inference_read_noise}")
    print(f"g_min: {args.gmin}")
    print(f"g_max: {args.gmax}")

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
    rpu_config.noise_model = PCMLikeNoiseModel(
        prog_noise_scale = 1.0,
        read_noise_scale = 1.0,
        drift_scale = 10.0,
        g_converter=SinglePairConductanceConverter(g_min=0, g_max=25),
        )
    rpu_config.drift_compensation = GlobalDriftCompensation()
    print(rpu_config.noise_model)
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

model_path = True
analog_model = None
if(model_path):
    model_save_path = './model/lstm_fp.pt'
    pre_model = torch.load(model_save_path).to(DEVICE)
    pre_model.rnn.flatten_parameters()
    del pre_model.encoder
    pre_model.forward = types.MethodType(new_forward, pre_model)

    analog_model = convert_to_analog(pre_model, gen_rpu_config())
    analog_model.load_state_dict(
            torch.load('./model/hwa.th', map_location=DEVICE),
            load_rpu_config=False
        )
    analog_model.rnn.flatten_parameters()
    encoder =torch.load('./model/encoder.pt').to(DEVICE)
else:
    model_save_path = './model/lstm_fp.pt'
    pre_model = torch.load(model_save_path).to(DEVICE)
    pre_model.rnn.flatten_parameters()
    analog_model = convert_to_analog(pre_model, gen_rpu_config())
    
#Since in the forward, it will perform log_softmax() on the output 
#Therefore, using NLLLoss here is equivalent to CrossEntropyLoss
criterion = nn.NLLLoss()

def evaluate(analog_model,data_source, analog):
    # Turn on evaluation mode which disables dropout.
    analog_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = analog_model.init_hidden(20)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = analog_model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                if(analog):
                    data = encoder(data)
                output, hidden = analog_model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


print('=' * 89)
print("Inference")
print('-' * 89)
start_time = 60
max_inference_time = 31536000
n_times = 9
t_inference_list = [
        0.0] + logspace(0, log10(float(max_inference_time)), n_times).tolist()
try:
    analog_model.eval()
    #t_inference in second
    for i, t_inference in enumerate(t_inference_list):
        analog_model.drift_analog_weights(t_inference)
        inference_loss = evaluate(analog_model, test_data, model_path)
        print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
        t_inference,inference_loss, math.exp(inference_loss)))
    print('=' * 89)
    print()
except KeyboardInterrupt:
    print('=' * 89)
    print('Exiting from Inference early')