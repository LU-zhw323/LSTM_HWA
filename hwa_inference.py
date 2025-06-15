import csv
import math
import os
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
from utils import compute_norm_accuracy
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_model.pt"
ENCODER_CHECKPOINT_PATH = "checkpoints/encoder.pt"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"
RESULTS_DIR = "results"





def create_csv_writer(csv_path):
    """
    Create a CSV writer for the given path.
    If the file exists, open it in append mode.
    If the file does not exist, create it and write the header.
    """
    file_exists = os.path.exists(csv_path)
    
    if file_exists:
        file_obj = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(file_obj)
    else:
        # create new file and write header
        file_obj = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(file_obj)
        
        header = ['t_inference', 'noise_scale', 'drift_scale', 'g_min', 'g_max', 
                  'memory_window', 'loss', 'perplexity', 'accuracy', 'error', 'norm_accuracy', 'norm_error']
        csv_writer.writerow(header)
    
    return file_obj, csv_writer, file_exists



def save_experiment_result(csv_writer, t_inference, noise_scale, drift_scale, 
                          g_min, g_max, loss, perplexity, accuracy, error_rate, norm_accuracy, norm_error):
    """
    Save an experiment result to a CSV file.
    
    Args:
        csv_writer: CSV writer
        t_inference: inference time
        noise_scale: noise scale
        drift_scale: drift scale
        g_min: minimum conductance
        g_max: maximum conductance
        loss: loss
        perplexity: perplexity
        accuracy: accuracy
        error_rate: error rate
        norm_accuracy: normalized accuracy
        norm_error: normalized error
    """
    memory_window = g_max - g_min
    row = [t_inference, noise_scale, drift_scale, g_min, g_max, 
           memory_window, loss, perplexity, accuracy, error_rate, norm_accuracy, norm_error]
    csv_writer.writerow(row)





def main():
    # setup rpu config
    lstm_config = LSTM_HWA_INFERENCE_Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # baseline
    fp_error = lstm_config.fp_error

    # gmax
    g_max = lstm_config.g_max

    # load data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, lstm_config.batch_size, lstm_config.seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)

    # time label
    time_mapping = {
        1: 'second',
        3600: 'hour',
        3600 * 24: 'day',
        3600 * 24 * 7: 'week',
        3600 * 24 * 365: 'year'
    }
    total_all_experiments = len(lstm_config.inference_time) * len(lstm_config.noise_scale) * \
                           len(lstm_config.drift_scale) * len(lstm_config.g_min)
    print(f"\nTotal experiments to run: {total_all_experiments}")
    
    # inference
    for t_inference in lstm_config.inference_time:
        t_label = time_mapping[t_inference]
        csv_filename = f"inference_results_{t_label}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_filename)
        file_obj, csv_writer, file_exists = create_csv_writer(csv_path)
        print(f"\n{'Appending to' if file_exists else 'Creating'} {csv_filename} (t={t_inference}s)")


        total_experiments = (len(lstm_config.noise_scale) * 
                           len(lstm_config.drift_scale) * 
                           len(lstm_config.g_min))
        
        pbar = tqdm(total=total_experiments, 
                   desc=f"T={t_label.upper()} experiments")
        # loop over all noise scales, drift scales, and g_min values
        try:
            # loop over all noise scales, drift scales, and g_min values
            for noise_scale in lstm_config.noise_scale:
                for drift_scale in lstm_config.drift_scale:
                    for g_min in lstm_config.g_min:
                        # update progress bar description
                        pbar.set_description(f"T={t_label.upper()}, noise={noise_scale}, "
                                           f"drift={drift_scale}, g_min={g_min}")
                        
                        rpu_config = hwa_rpu_config(
                            hwa_noise_scale=lstm_config.hwa_noise_scale,
                            hwa_pdrop=lstm_config.pdrop,
                            noise_scale=noise_scale,
                            drift_scale=drift_scale,
                            g_min=g_min,
                            g_max=g_max,
                        )
                        
                        # load model
                        hwa_model, fp_embedding_layer = load_hwa_model_and_encoder(
                            HWA_CHECKPOINT_PATH, ENCODER_CHECKPOINT_PATH, 
                            vocab_size, lstm_config, rpu_config, DEVICE, False)
                        
                        
                        # evaluate hwa model
                        test_loss, test_perplexity, test_accuracy, test_error_rate = inference_hwa(
                            hwa_model, fp_embedding_layer, test_data, vocab_size, 
                            t_inference, lstm_config.num_evals, DEVICE)
                        
                         # get normalized error rate
                        norm_accuracy = compute_norm_accuracy(vocab_size, fp_error, test_error_rate)
                        norm_error = 1.0 - norm_accuracy
                        
                        # save results to CSV
                        save_experiment_result(csv_writer, t_inference, noise_scale, 
                                             drift_scale, g_min, g_max, test_loss, 
                                             test_perplexity, test_accuracy, test_error_rate, norm_accuracy, norm_error)
                        
                        # flush file buffer to ensure data is written
                        file_obj.flush()
                        
                        # print current results
                        '''print(f"\nResults: Loss={test_loss:.4f}, Perplexity={test_perplexity:.4f}, "
                              f"Accuracy={test_accuracy:.4f}, Error={test_error_rate:.4f}")'''
                        
                        # update progress bar
                        pbar.update(1)
                        
        finally:
            # ensure file is closed
            pbar.close()
            file_obj.close()
            print(f"\nResults saved to {csv_path}")
    
    print("\nAll experiments completed!")
                    
        



if __name__ == "__main__":
    main()