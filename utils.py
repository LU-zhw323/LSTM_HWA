import numpy as np
import torch
from data import Dictionary, Corpus, SequentialBatcher
from torch.nn import functional as F
import math
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
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter


def gen_rpu_config(
        hwa_noise_scale: float=5.0,
        hwa_pdrop: float=0.01,
        noise_scale: float=1.0, 
        drift_scale: float=1.0, 
        g_min: float=0.0, 
        g_max: float=25.0
    )->InferenceRPUConfig:
    """
    Generate a rpu config for hwa training
    Args:
        hwa_noise_scale: noise scale for hwa training
        hwa_pdrop: dropout rate for hwa training
        noise_scale: noise scale for noise model
        drift_scale: drift scale for noise model
        g_min: minimum conductance for noise model
        g_max: maximum conductance for noise model
    Returns:
        rpu_config: a rpu config for hwa training
    """
    rpu_config = InferenceRPUConfig()

    # modifier for hwa training
    rpu_config.modifier.type = WeightModifierType.PCM_NOISE
    rpu_config.modifier.std_dev = hwa_noise_scale
    rpu_config.modifier.pdrop = hwa_pdrop
    rpu_config.modifier.pcm_t0 = 20

    # weight clipping
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.5 

    # forward
    rpu_config.forward.out_res = 8
    rpu_config.forward.inp_res = 8
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.out_bound = 10.0 #1.0 if rpu_config.mapping.learn_out_scaling = True
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
    #rpu_config.mapping.learn_out_scaling = True

    # learn input range
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.learn_input_range = True
    rpu_config.pre_post.input_range.init_std_alpha = 1.0
    rpu_config.pre_post.input_range.decay = 0.001
    rpu_config.pre_post.input_range.gradient_relative = True
    rpu_config.pre_post.input_range.gradient_scale = 1.0
    rpu_config.pre_post.input_range.init_from_data = 1000
    rpu_config.pre_post.input_range.input_min_percentage = 0.95
    rpu_config.pre_post.input_range.manage_output_clipping = True  
    rpu_config.pre_post.input_range.output_min_percentage = 0.95
    
    

    # noise model
    rpu_config.noise_model = PCMLikeNoiseModel(
        g_max=g_max,
        prog_noise_scale=noise_scale,
        read_noise_scale=noise_scale,
        drift_scale=drift_scale,
        g_converter=SinglePairConductanceConverter(g_max=g_max, g_min=g_min),
    )
    rpu_config.drift_compensation = GlobalDriftCompensation()

    return rpu_config

def setup_data(data_path, batch_size, seq_length):
    # create corpus
    corp = Corpus(data_path)

    # create sequential batcher
    train_data = SequentialBatcher(corp.train, batch_size, seq_length)
    valid_data = SequentialBatcher(corp.valid, batch_size, seq_length)
    test_data = SequentialBatcher(corp.test, batch_size, seq_length)

    return train_data, valid_data, test_data, corp



def adjust_learning_rate(optimizer, epoch, init_lr=1.0, lr_decay_start=6, lr_decay_factor=1.2):
    # decay learning rate
    if epoch >= lr_decay_start:
        lr = init_lr / (lr_decay_factor ** (epoch - lr_decay_start))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = init_lr
    return lr


@torch.no_grad()
def evaluate_fp(model, data_loader, vocab_size, device):
    """
    Evaluate the model on the data loader for fp training
    Args:
        model: the model to evaluate
        data_loader: the data loader to evaluate on
        vocab_size: the size of the vocabulary
        device: the device to evaluate on
    Returns:
        avg_loss: the average loss
        perplexity: the perplexity
        accuracy: the accuracy
        error_rate: the error rate
    """
    
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        hidden = model.init_hidden(data_loader.batch_size, device)
        
        for i in range(num_batches):
            inputs, targets = data_loader.get_batch(i)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            output, hidden = model(inputs, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            loss = F.cross_entropy(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=-1)
            predictions_flat = predictions.view(-1)
            targets_flat = targets.view(-1)
            
            correct = (predictions_flat == targets_flat).sum().item()
            total_correct += correct
            total_predictions += targets_flat.size(0)
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_predictions
    error_rate = 1 - accuracy
    return avg_loss, perplexity, accuracy, error_rate



def save_checkpoint(model, optimizer, epoch, loss, perplexity, filepath="checkpoints/model.pt"):
    # save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
    }
    torch.save(checkpoint, filepath)



def load_checkpoint(filepath, model, optimizer):
    # load checkpoint
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['perplexity']


def compute_norm_error(vocab_size: int, fp_error: float, hwa_error: float):
    # compute chance error
    error_chance = 1.0 - 1.0 / vocab_size

    # compute norm error
    return  1.0 - (hwa_error - fp_error) / (error_chance - fp_error)
