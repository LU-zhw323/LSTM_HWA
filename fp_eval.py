import math
from data import Dictionary, Corpus
from utils import save_checkpoint, setup_data, adjust_learning_rate, evaluate_fp, load_checkpoint
import torch
from lstm import LSTM_PTB
from torch.nn import functional as F
from tqdm import tqdm

DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/model.pt"


def main():
    

    # hyper parameters
    embedding_dim = 650
    hidden_size = 650
    num_layers = 2
    dropout = 0.5
    batch_size = 20
    seq_length = 35
    lr = 20.0
    lr_decay_start = 20 
    lr_decay_factor = 1.2
    max_grad_norm = 0.25
    epochs = 40

    # setup data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, batch_size, seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)

    # load model
    model = LSTM_PTB(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(DEVICE)
    load_checkpoint(CHECKPOINT_PATH, model, None)
    model.eval()

    # evaluate on test set
    test_loss, test_perplexity, test_accuracy, test_error_rate = evaluate_fp(model, test_data, vocab_size, DEVICE)
    print(f"Test Loss: {test_loss:.3f} | Test Perplexity: {test_perplexity:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")

if __name__ == "__main__":
    main()