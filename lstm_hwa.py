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
import utils
###############################################################################
# Prepare file
###############################################################################


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
initial_lr = 0

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
    args.lr = 0.01
    global initial_lr
    initial_lr = args.lr
    print(f"Learning Rate: {args.lr}")

    args.dropout = 0.5
    print(f"Dropout Rate: {args.dropout}")

    args.noise = 3.4
    print(f"Noise: {args.noise}")

    args.clip = 10
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


def create_sgd_optimizer(model):
    
    optimizer = AnalogSGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.w_decay)
    optimizer.regroup_param_groups(model)

    return optimizer

###############################################################################
# Build the model
###############################################################################
model_save_path = './model/lstm_fp.pt'
pre_model = torch.load(model_save_path).to(DEVICE)
pre_model.rnn.flatten_parameters()
model = convert_to_analog(pre_model, gen_rpu_config())
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
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
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
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), f"./model/lstm_hwa_{args.noise}.th")
            best_val_loss = val_loss
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

model.load_state_dict(torch.load(f"./model/lstm_hwa_{args.noise}.th", map_location=DEVICE))
model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# Write to file
with h5py.File('./result/lstm_hwa.h5', 'a') as f:
    task_group = f.require_group(f"task_noise_{args.noise}")
    if 'train_results' in task_group:
            del task_group['train_results']
    task_group.create_dataset('train_results', data=test_loss)


print()
utils.inference(model, evaluate, test_data, args, './result/lstm_hwa.h5', f"task_noise_{args.noise}")