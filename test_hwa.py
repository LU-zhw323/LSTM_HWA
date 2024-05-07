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
import types
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

from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
import data
import csv
import utils
###############################################################################
# Prepare file
###############################################################################


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
initial_lr = 0


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
    global initial_lr
    initial_lr = args.lr
    print(f"Learning Rate: {args.lr}")

    args.dropout = 0.5
    print(f"Dropout Rate: {args.dropout}")

    args.noise = 5.0
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


###############################################################################
# Load Data
###############################################################################
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
        g_converter=SinglePairConductanceConverter(g_min=0, g_max=25),
        prog_noise_scale = 1.0,
        read_noise_scale = 1.0,
        drift_scale = 0.0
        )
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


def create_sgd_optimizer(model):
    
    optimizer = AnalogSGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.w_decay)
    optimizer.regroup_param_groups(model)

    return optimizer

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

model_save_path = './model/lstm_fp.pt'
pre_model = torch.load(model_save_path).to(DEVICE)
pre_model.rnn.flatten_parameters()

###Detach Encoder
encoder = pre_model.encoder
del pre_model.encoder
pre_model.forward = types.MethodType(new_forward, pre_model)

rpu = gen_rpu_config(args)
model = convert_to_analog(pre_model, rpu)
model.rnn.flatten_parameters()
optimizer = create_sgd_optimizer(model)
#Since in the forward, it will perform log_softmax() on the output 
#Therefore, using NLLLoss here is equivalent to CrossEntropyLoss
criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_decay, patience=0, verbose=False)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                data = encoder(data)
                output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_len)): 
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            data = encoder(data)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2g} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.seq_len, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    
# Loop over epochs.
#lr = args.lr
best_val_loss = None

print()
print('=' * 89)
print("Training")
print('-' * 89)
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        scheduler.step(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))

print('=' * 89)
torch.save(model.state_dict(), "./hwa.th")
torch.save(encoder, './encoder_module.pt')

print()
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
    model.eval()
    #t_inference in second
    for i, t_inference in enumerate(t_inference_list):
        model.drift_analog_weights(t_inference)
        inference_loss = evaluate(test_data)
        print('| Inference | time {} | test loss {:5.2f} | test ppl {:8.2f}'.format(
        t_inference,inference_loss, math.exp(inference_loss)))
        inference_data[i] = (args.noise, t_inference, inference_loss, math.exp(inference_loss))
    print('=' * 89)
    print()
except KeyboardInterrupt:
    print('=' * 89)
    print('Exiting from Inference early')