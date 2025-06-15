import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from tqdm import tqdm
from hwa_utils import covert_fp_to_hwa, evaluate_hwa, inference_hwa, load_hwa_model_and_encoder, save_hwa_model, train_step_hwa, warmup_hwa
from lstm import LSTM_PTB
from utils import load_checkpoint, setup_data
from hwa_rpu import hwa_rpu_config
from config import LSTM_HWA_INFERENCE_Config
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from utils import set_seed
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_model.pt"
ENCODER_CHECKPOINT_PATH = "checkpoints/encoder.pt"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"


def main():
    # setup rpu config
    lstm_config = LSTM_HWA_INFERENCE_Config()
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=lstm_config.hwa_noise_scale,
        hwa_pdrop=lstm_config.pdrop,
        noise_scale=lstm_config.noise_scale,
        drift_scale=lstm_config.drift_scale,
        g_min=lstm_config.g_min,
        g_max=lstm_config.g_max,
    )
    
    # load data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, lstm_config.batch_size, lstm_config.seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)
    
    # load model
    hwa_model, fp_embedding_layer = load_hwa_model_and_encoder(HWA_CHECKPOINT_PATH, ENCODER_CHECKPOINT_PATH, vocab_size, lstm_config, rpu_config, DEVICE, True)
    
    # inference
    for t_inference in lstm_config.inference_time:
        # evaluate hwa model
        test_loss, test_perplexity, test_accuracy, test_error_rate = inference_hwa(
            hwa_model, fp_embedding_layer, test_data, vocab_size, t_inference, lstm_config.num_evals, DEVICE)
        print("-" * 80)
        print(f"Inference Time: {t_inference} | Test Loss: {test_loss:.3f} | Test Perplexity: {test_perplexity:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
        print("-" * 80)



if __name__ == "__main__":
    main()