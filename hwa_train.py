import argparse
import torch
from tqdm import tqdm
from hwa_utils import covert_fp_to_hwa, evaluate_hwa, inference_hwa, load_hwa_model_and_encoder, save_hwa_model, train_step_hwa
from lstm import LSTM_PTB
from utils import load_checkpoint, setup_data
from hwa_rpu import hwa_rpu_config
from config import LSTM_HWA_Config
from aihwkit.optim import AnalogSGD
from utils import set_seed
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_model.pt"
ENCODER_CHECKPOINT_PATH = "checkpoints/encoder.pt"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"
HWA_FINAL_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"


def parse_args():
    """Command-line options of one HWA training run."""
    parser = argparse.ArgumentParser(description="HWA fine-tuning of the FP LSTM on PTB.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python, NumPy and PyTorch.")
    parser.add_argument("--run_id", type=int, default=None,
                        help="Save to checkpoints/hwa_model_<run_id>.th and hwa_model_final_<run_id>.th "
                             "instead of hwa_model.th and hwa_model_final.th.")
    return parser.parse_args()


def main():
    """Fine-tunes the analog LSTM and output layers of `FP_CHECKPOINT_PATH` with PCM weight noise.

    Saves the lowest-validation-error model and the last-epoch model, and the frozen embedding to
    `ENCODER_CHECKPOINT_PATH`. Then prints the test metrics of both models at
    `LSTM_HWA_Config.t_inference`. Overwrites existing checkpoints at these paths.
    """
    args = parse_args()
    if args.run_id is None:
        best_checkpoint_path, final_checkpoint_path = HWA_CHECKPOINT_PATH, HWA_FINAL_CHECKPOINT_PATH
    else:
        best_checkpoint_path = f"checkpoints/hwa_model_{args.run_id}.th"
        final_checkpoint_path = f"checkpoints/hwa_model_final_{args.run_id}.th"

    # set seed
    set_seed(args.seed)
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
    load_checkpoint(FP_CHECKPOINT_PATH, model, None)

    # convert fp model to hwa model
    fp_embedding_layer, hwa_model = covert_fp_to_hwa(model, rpu_config, DEVICE)

    # optimizer
    optimizer = AnalogSGD(hwa_model.parameters(), lr=lstm_config.lr, momentum=lstm_config.momentum, weight_decay=lstm_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lstm_config.lr_decay_factor, patience=0)
   
    # train hwa model
    best_valid_error_rate = float('inf')
    for epoch in tqdm(range(lstm_config.epochs), desc="Training"):
        current_lr = optimizer.param_groups[0]['lr']
        # train hwa model
        train_loss, train_perplexity = train_step_hwa(
            hwa_model, fp_embedding_layer, train_data, vocab_size, optimizer, lstm_config.max_grad_norm, DEVICE)
        print("-" * 80)
        print(f"Epoch {epoch+1:2d} | Lr: {current_lr:.3f} | Train Loss: {train_loss:.3f} | Train Perplexity: {train_perplexity:.3f}")
        # evaluate hwa model
        valid_loss, valid_perplexity, valid_accuracy, valid_error_rate = evaluate_hwa(
            hwa_model, fp_embedding_layer, valid_data, vocab_size, lstm_config.num_evals, DEVICE)
        
        print(f"Epoch {epoch+1:2d} | Lr: {current_lr:.3f} | Valid Loss: {valid_loss:.3f} | Valid Perplexity: {valid_perplexity:.3f} | Valid Accuracy: {valid_accuracy:.3f} | Valid Error Rate: {valid_error_rate:.3f}")
        print("-" * 80)
        scheduler.step(valid_error_rate)
        if valid_error_rate < best_valid_error_rate:
            best_valid_error_rate = valid_error_rate
            save_hwa_model(hwa_model, fp_embedding_layer, best_checkpoint_path, ENCODER_CHECKPOINT_PATH)

    # save final hwa model
    save_hwa_model(hwa_model, fp_embedding_layer, final_checkpoint_path, ENCODER_CHECKPOINT_PATH)
    test_loss, test_perplexity, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, fp_embedding_layer, test_data, vocab_size, lstm_config.t_inference, lstm_config.num_evals, DEVICE)
    print("-" * 80)
    print(f"FINAL | Test Loss: {test_loss:.3f} | Test Perplexity: {test_perplexity:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
    print("-" * 80)

    
    # load best hwa model
    hwa_model, fp_embedding_layer = load_hwa_model_and_encoder(best_checkpoint_path, ENCODER_CHECKPOINT_PATH, vocab_size, lstm_config, rpu_config, DEVICE, True)
    # evaluate hwa model
    test_loss, test_perplexity, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, fp_embedding_layer, test_data, vocab_size, lstm_config.t_inference, lstm_config.num_evals, DEVICE)
    print("-" * 80)
    print(f"BEST | Test Loss: {test_loss:.3f} | Test Perplexity: {test_perplexity:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
    print("-" * 80)


if __name__ == "__main__":
    main()