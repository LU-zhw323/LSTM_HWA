import random
import numpy as np
import torch
from data import Corpus, SequentialBatcher
from torch.nn import functional as F
import math


def setup_data(data_path, batch_size, seq_length):
    """PTB train, valid and test batchers built from the corpus in `data_path`.

    Returns:
        (train_data, valid_data, test_data, corpus). The batchers are `SequentialBatcher`s.
        The vocabulary is `corpus.dictionary`.
    """
    # create corpus
    corp = Corpus(data_path)

    # create sequential batcher
    train_data = SequentialBatcher(corp.train, batch_size, seq_length)
    valid_data = SequentialBatcher(corp.valid, batch_size, seq_length)
    test_data = SequentialBatcher(corp.test, batch_size, seq_length)

    return train_data, valid_data, test_data, corp


@torch.no_grad()
def evaluate_fp(model, data_loader, vocab_size, device):
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

            output, hidden = model(inputs, hidden)
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



def save_checkpoint(model, optimizer, epoch, loss, perplexity, filepath="checkpoints/model.pt"):
    """Saves the model and optimizer state with the epoch, loss and perplexity to `filepath`."""
    # save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
    }
    torch.save(checkpoint, filepath)



def load_checkpoint(filepath, model, optimizer):
    """Loads a `save_checkpoint` file into `model`, and into `optimizer` unless it is None.

    Returns:
        (epoch, loss, perplexity) stored in the checkpoint.
    """
    # load checkpoint
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['perplexity']


def compute_norm_accuracy(vocab_size: int, fp_error: float, hwa_error: float):
    """Normalized accuracy of an HWA model: 1 at the FP error rate, 0 at the chance error rate 1 - 1/vocab_size.

    Errors are fractions in [0, 1]. The result exceeds 1 when `hwa_error` is below `fp_error`.
    """
    # compute chance error
    error_chance = 1.0 - 1.0 / vocab_size

    # compute norm error
    return  1.0 - (hwa_error - fp_error) / (error_chance - fp_error)


def set_seed(seed=42):
    """Seeds the Python, NumPy and PyTorch (CPU and all CUDA devices) random generators.

    Also sets cuDNN to deterministic, non-benchmark mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
