import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from lstm_model import LSTM_PTB
from utils import gen_rpu_config, load_checkpoint, setup_data
from lstm_config import LSTM_HWA_Config
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/model.pt"




def evaluate(model, encoder, data_loader, vocab_size, device):
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

            # encode inputs
            embedded_inputs = encoder(inputs)
            
            output, hidden = model.forward_lstm_only(embedded_inputs, hidden)
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
    rpu_config = gen_rpu_config(
        hwa_noise_scale=lstm_config.hwa_noise_scale,
        hwa_pdrop=lstm_config.hwa_pdrop,
    )

    # load data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, lstm_config.batch_size, lstm_config.seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)

    # load model
    model = LSTM_PTB(vocab_size, lstm_config.embedding_dim, lstm_config.hidden_size, lstm_config.num_layers, lstm_config.dropout).to(DEVICE)
    load_checkpoint(CHECKPOINT_PATH, model, None)
    model.eval()

    # get fp embedding layer
    fp_embedding_layer = model.get_embedding_component().to(DEVICE)

    # evaluate
    test_loss, test_perplexity, test_accuracy, test_error_rate = evaluate(model, fp_embedding_layer, test_data, vocab_size, DEVICE)
    print(f"Test Loss: {test_loss:.3f} | Test Perplexity: {test_perplexity:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")



if __name__ == "__main__":
    main()