import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from hwa_utils import inference_hwa, load_hwa_model_and_encoder
from lstm import LSTM_PTB
from utils import compute_norm_accuracy, load_checkpoint, setup_data
from hwa_rpu import direct_mapping_rpu_config, hwa_rpu_config
from config import LSTM_HWA_Config
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_model.pt"
ENCODER_CHECKPOINT_PATH = "checkpoints/encoder.pt"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"



def direct_mapping_hwa(model, encoder, train_data, test_data, vocab_size, device, t_inference):
    """
    Evaluate the model on the data loader for fp training
    Args:
        model: the model to evaluate
        data_loader: the data loader to evaluate on
        vocab_size: the size of the vocabulary
        device: the device to evaluate on
        t_inference: the time of inference
    Returns:
        avg_loss: the average loss
        perplexity: the perplexity
        accuracy: the accuracy
        error_rate: the error rate
    """
    # use hwa training flow to map the model to hwa
    model.train()
    lr = 0.0
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.requires_grad = False
    hidden = model.init_hidden(train_data.batch_size, DEVICE)
    for i in range(1000):
        inputs, targets = train_data.get_batch(i)
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
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
        # update weights
        optimizer.step()

    
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    num_batches = len(test_data)
    
    with torch.no_grad():
        
        model.drift_analog_weights(t_inference)
        hidden = model.init_hidden(test_data.batch_size, device)
        for i in range(num_batches):
            inputs, targets = test_data.get_batch(i)
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
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_predictions
    error_rate = 1 - accuracy
    return avg_loss, perplexity, accuracy, error_rate



def main():

    # setup rpu config
    lstm_config = LSTM_HWA_Config()
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=lstm_config.hwa_noise_scale,
        hwa_pdrop=lstm_config.pdrop,
        noise_scale=1.0,
        drift_scale=1.0,
        g_min=0.0,
        g_max=25.0,
    )
    time_mapping = {
        1: 'second',
        3600: 'hour',
        3600 * 24: 'day',
        3600 * 24 * 7: 'week',
        3600 * 24 * 365: 'year'
    }

    # load data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, lstm_config.batch_size, lstm_config.seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)

    # load model
    hwa_model, fp_embedding_layer = load_hwa_model_and_encoder(
        HWA_CHECKPOINT_PATH, ENCODER_CHECKPOINT_PATH, 
        vocab_size, lstm_config, rpu_config, DEVICE, True)

    # baseline
    fp_error = 0.72794

    # evaluate
    t_inferences = [1, 3600, 3600*24, 3600*24*7, 3600*24*365]
    for t_inference in t_inferences:
        t_label = time_mapping[t_inference]
        test_loss, test_perplexity, test_accuracy, test_error_rate = inference_hwa(
            hwa_model, fp_embedding_layer, test_data, vocab_size, 
            t_inference, lstm_config.num_evals, DEVICE)
        
        # get normalized error rate
        norm_accuracy = compute_norm_accuracy(vocab_size, fp_error, test_error_rate)
        norm_error = 1.0 - norm_accuracy
        print('-'*100)
        print(f"T={t_label.upper()}, Loss={test_loss:.4f}, Perplexity={test_perplexity:.4f}, Accuracy={test_accuracy:.4f}, Error={test_error_rate:.4f}, Norm Accuracy={norm_accuracy:.4f}, Norm Error={norm_error:.4f}")
        print('-'*100)

if __name__ == "__main__":
    main()