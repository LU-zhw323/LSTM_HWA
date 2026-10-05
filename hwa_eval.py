import torch
from hwa_utils import inference_hwa, load_hwa_model_and_encoder
from utils import compute_norm_accuracy, setup_data
from hwa_rpu import hwa_rpu_config
from config import LSTM_HWA_Config
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_CHECKPOINT_PATH = "checkpoints/encoder.pt"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"



def main():
    """Prints test metrics of `HWA_CHECKPOINT_PATH` at 1 s, 1 h, 1 day, 1 week and 1 year after programming.

    Uses the RPU config stored in the checkpoint and averages `LSTM_HWA_Config.num_evals` noise draws per time.
    """

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