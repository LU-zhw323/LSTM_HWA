import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
    

class SequentialBatcher:
    
    def __init__(self, data, batch_size, seq_len):
        self.seq_len = seq_len
        self.batch_size = batch_size
        num_seqs = len(data) // batch_size
        data = data[:num_seqs * batch_size]
        self.data = data.view(batch_size, -1)
        self.num_batches = (self.data.size(1) - 1) // seq_len

    def get_batch(self, i):
        seq_len = min(self.seq_len, self.data.size(1) - 1 - i * self.seq_len)
        start_idx = i * self.seq_len
        end_idx = start_idx + seq_len
        data = self.data[:, start_idx:end_idx]
        target = self.data[:, start_idx+1:end_idx+1]
        return data, target
    
    def __len__(self):
        return self.num_batches