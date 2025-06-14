import math
from data import Dictionary, Corpus
from utils import load_checkpoint, save_checkpoint, set_seed, setup_data, adjust_learning_rate, evaluate_fp
import torch
from lstm import LSTM_PTB
from torch.nn import functional as F
from tqdm import tqdm
from config import LSTM_FP_Config

DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/fp_model.pt"







def main():
    

    # hyper parameters
    config = LSTM_FP_Config()
    embedding_dim = config.embedding_dim
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    dropout = config.dropout
    batch_size = config.batch_size
    seq_length = config.seq_length
    lr = config.lr
    max_grad_norm = config.max_grad_norm
    epochs = config.epochs
    
    # set seed
    set_seed()

    # setup data
    train_data, valid_data, test_data, corp = setup_data(DATA_PATH, batch_size, seq_length)
    vocab_size = len(corp.dictionary)

    # get number of batches
    num_train_batches = len(train_data)
    num_valid_batches = len(valid_data)
    num_test_batches = len(test_data)

    # model
    model = LSTM_PTB(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(DEVICE)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=config.weight_decay)



    # training loop
    best_valid_loss = float('inf')
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        # adjust learning rate
        #current_lr = adjust_learning_rate(optimizer, epoch, lr, lr_decay_start, lr_decay_factor)

        hidden = model.init_hidden(batch_size, DEVICE)

        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']

        for i in range(0, num_train_batches):
            inputs, targets = train_data.get_batch(i)
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            output,hidden = model(inputs,hidden)
            # detach hidden states
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = F.cross_entropy(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # update weights
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                perplexity = math.exp(avg_loss)
                print(f"  Epoch {epoch+1:2d} | Batch {i+1:4d}/{len(train_data)} | "
                      f"Loss: {avg_loss:.3f} | PPL: {perplexity:.2f} | LR: {current_lr:.6f}")

        avg_train_loss = total_loss / len(train_data)
        train_perplexity = math.exp(avg_train_loss)

        # evaluate on validation set
        valid_loss, valid_perplexity, valid_accuracy, valid_error_rate = evaluate_fp(model, valid_data, vocab_size, DEVICE)

        print("-" * 80)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f}")
        print(f"  Train Loss: {avg_train_loss:.3f} | Train PPL: {train_perplexity:.2f}")
        print(f"  Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_perplexity:.2f}")
        print(f"  Valid Accuracy: {valid_accuracy:.2f} | Valid Error Rate: {valid_error_rate:.2f}")
        print("-" * 80)
        # save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, valid_loss, valid_perplexity, 
                          CHECKPOINT_PATH)
            print(f"Best loss: {valid_loss:.2f}")
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= config.lr_decay_factor
        
        # save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, valid_loss, valid_perplexity,
                          f"checkpoints/fp_checkpoint_epoch_{epoch+1}.pt")
    print("-" * 80)
    print("Training complete")
    
    # evaluate on test set
    # load best model
    load_checkpoint(CHECKPOINT_PATH, model, None)
    test_loss, test_perplexity, test_accuracy, test_error_rate = evaluate_fp(model, test_data, vocab_size, DEVICE)
    print(f"Test Loss: {test_loss:.3f} | Test PPL: {test_perplexity:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f} | Test Error Rate: {test_error_rate:.2f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
