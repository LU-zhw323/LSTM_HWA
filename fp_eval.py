from utils import setup_data, evaluate_fp, load_checkpoint
import torch
from lstm import LSTM_PTB
from config import LSTM_FP_Config

DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/fp_model.pt"


def main():
    """Prints the test loss, perplexity, accuracy and error rate of the FP model at `CHECKPOINT_PATH`."""

    # hyper parameters
    config = LSTM_FP_Config()

    # setup data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, config.batch_size, config.seq_length)
    vocab_size = len(corp.dictionary)

    # load model
    model = LSTM_PTB(vocab_size, config.embedding_dim, config.hidden_size, config.num_layers, config.dropout).to(DEVICE)
    load_checkpoint(CHECKPOINT_PATH, model, None)
    model.eval()

    # evaluate on test set
    test_loss, test_perplexity, test_accuracy, test_error_rate = evaluate_fp(model, test_data, vocab_size, DEVICE)
    print(f"Test Loss: {test_loss:.5f} | Test Perplexity: {test_perplexity:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f} | Test Error Rate: {test_error_rate:.5f}")

if __name__ == "__main__":
    main()
