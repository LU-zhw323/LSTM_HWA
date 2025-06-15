from typing import Tuple
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
import math
import numpy as np
import torch
import torch.nn.functional as F
from aihwkit.nn.conversion import convert_to_analog
import torch.nn as nn

from lstm import LSTM_PTB, AnalogLSTM_PTB
from torch.serialization import add_safe_globals


def train_step_hwa(model, encoder, train_data, vocab_size, optimizer, max_grad_norm, device)->Tuple[float, float]:
    """
    Train the model on the data loader for hwa training
    Args:
        model: the model to train
        encoder: the encoder to use
        train_data: the data loader to train on
        vocab_size: the size of the vocabulary
        device: the device to train on
        optimizer: the optimizer to use
        max_grad_norm: the maximum gradient norm
    """
    model.train()
    train_loss = 0.0
    num_batches = len(train_data)
    hidden = model.init_hidden(train_data.batch_size, device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False  # freeze embedding layer

    for i in range(num_batches):
        inputs, targets = train_data.get_batch(i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        # zero gradients
        optimizer.zero_grad()
        # encode inputs
        embedded_inputs = encoder(inputs)
        # forward pass
        output, hidden = model(embedded_inputs, hidden)
        # detach hidden states
        hidden = (hidden[0].detach(), hidden[1].detach())
        loss = F.cross_entropy(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # update weights
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / num_batches
    avg_train_perplexity = math.exp(avg_train_loss)
    return avg_train_loss, avg_train_perplexity


@torch.no_grad()
def evaluate_hwa(model, encoder, data_loader, vocab_size, t_inference, num_evals, device):
    """
    Evaluate the model on the data loader for hwa training
    Args:
        model: the model to evaluate
        encoder: the encoder to use
        data_loader: the data loader to evaluate on
        vocab_size: the size of the vocabulary
        t_inference: the time of inference
        num_evals: the number of evaluations
        device: the device to evaluate on
    Returns:
        avg_loss: the average loss
        perplexity: the perplexity
        accuracy: the accuracy
        error_rate: the error rate
    """
    model.eval()
    encoder.eval()
    all_losses = []
    all_accuracies = []
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for _ in range(num_evals):
            total_loss = 0.0
            total_correct = 0.0
            total_predictions = 0.0
            # drift analog weights
            model.drift_analog_weights(t_inference)
            hidden = model.init_hidden(data_loader.batch_size, device)
            for i in range(num_batches):
                inputs, targets = data_loader.get_batch(i)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # encode inputs
                embedded_inputs = encoder(inputs)
                output, hidden = model(embedded_inputs, hidden)
                
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                loss = F.cross_entropy(output.view(-1, vocab_size), targets.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(output, dim=-1)
                predictions_flat = predictions.view(-1)
                targets_flat = targets.view(-1)
                
                correct = (predictions_flat == targets_flat).sum().item()
                total_correct += correct
                total_predictions += targets_flat.size(0)
            trial_loss = total_loss / num_batches
            trial_accuracy = total_correct / total_predictions
            all_losses.append(trial_loss)
            all_accuracies.append(trial_accuracy)
    
    avg_loss = np.mean(all_losses)
    avg_accuracy = np.mean(all_accuracies)
    avg_error_rate = 1 - avg_accuracy
    avg_perplexity = math.exp(avg_loss)
    return avg_loss, avg_perplexity, avg_accuracy, avg_error_rate




def covert_fp_to_hwa(fp_model: nn.Module, rpu_config: InferenceRPUConfig, device: torch.device):
    """
    Convert the fp model to a hwa model
    Args:
        fp_model: the fp model to convert
        rpu_config: the rpu config to use
    """
    # get fp embedding layer
    fp_embedding_layer = fp_model.get_embedding_component().to(device)
    fp_embedding_layer.eval()
    for param in fp_embedding_layer.parameters():
        param.requires_grad = False  # freeze embedding layer
    
    # convert fp model to hwa model
    fp_lstm_layer, fp_dropout = fp_model.get_lstm_component()
    fp_fc_layer = fp_model.get_output_component()
    hwa_model = AnalogLSTM_PTB(fp_lstm_layer, fp_dropout, fp_fc_layer)
    analog_model = convert_to_analog(hwa_model, rpu_config).to(device)
    return fp_embedding_layer, analog_model



def save_hwa_model(analog_model, encoder, analog_model_path, encoder_path):
    """
    Save the hwa model
    Args:
        analog_model: the hwa model to save
        encoder: the encoder to save
        filepath: the path to save the model
    """
    torch.save(analog_model.state_dict(), analog_model_path)
    torch.save(encoder.state_dict(), encoder_path)


def load_hwa_model(config, vocab_size, analog_model_path, encoder_path, rpu_config, device, load_rpu=False):
    """
    Load the hwa model
    Args:
        analog_model_path: the path to load the hwa model
        encoder_path: the path to load the encoder
    """
    fp_model = LSTM_PTB(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout).to(device)
    encoder = fp_model.get_embedding_component().to(device)
    # convert fp model to hwa model
    fp_lstm_layer, fp_dropout = fp_model.get_lstm_component()
    fp_fc_layer = fp_model.get_output_component()
    hwa_model = AnalogLSTM_PTB(fp_lstm_layer, fp_dropout, fp_fc_layer)
    analog_model = convert_to_analog(hwa_model, rpu_config).to(device)

    # load hwa model
    analog_model.load_state_dict(
            torch.load(analog_model_path, map_location=device, weights_only=False),
            load_rpu_config=load_rpu
        )
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    analog_model.eval()
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False  # freeze embedding layer
    return analog_model, encoder