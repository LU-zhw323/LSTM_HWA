import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from tqdm import tqdm
from hwa_utils import evaluate_hwa, train_step_hwa
from lstm import LSTM_PTB
from utils import load_checkpoint, setup_data
from hwa_rpu import hwa_rpu_config
from config import LSTM_HWA_Config
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/model.pt"



def main():

    # setup rpu config
    lstm_config = LSTM_HWA_Config()
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=lstm_config.hwa_noise_scale,
        hwa_pdrop=lstm_config.pdrop,
        noise_scale=lstm_config.noise_scale,
        drift_scale=lstm_config.drift_scale,
        g_min=lstm_config.g_min,
        g_max=lstm_config.g_max
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

    # convert lstm to hwa
    hwa_model = convert_to_analog(model, rpu_config).to(DEVICE)

    # optimizer
    optimizer = AnalogSGD(hwa_model.parameters(), lr=lstm_config.lr, momentum=lstm_config.momentum, weight_decay=lstm_config.weight_decay)

    # train hwa model
    best_valid_error_rate = float('inf')
    for epoch in tqdm(range(lstm_config.epochs), desc="Training"):
        # train hwa model
        train_loss, train_perplexity = train_step_hwa(
            hwa_model, fp_embedding_layer, train_data, vocab_size, optimizer, lstm_config.max_grad_norm, DEVICE)

        # evaluate hwa model
        test_loss, test_perplexity, test_accuracy, test_error_rate = evaluate_hwa(
            hwa_model, fp_embedding_layer, valid_data, vocab_size, lstm_config.t_inference, lstm_config.num_evals, DEVICE)
        
        # 


if __name__ == "__main__":
    main()