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

    for i in range(num_batches):
        inputs, targets = train_data.get_batch(i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        # zero gradients
        optimizer.zero_grad()
        # encode inputs
        embedded_inputs = encoder(inputs)
        # forward pass
        lstm_out, hidden = model.forward_lstm_only(embedded_inputs, hidden)
        output = model.forward_output_only(lstm_out)
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
                
                lstm_out, hidden = model.forward_lstm_only(embedded_inputs, hidden)
                output = model.forward_output_only(lstm_out)
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