import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, filename=None, data=None, block_size=1):
        if filename is None and data is None:
            raise RuntimeError("chardataset, supplied data are both none, must specify at least one data source")

        if filename is not None:
            self.fn = filename

            self.data = open(self.fn,'r').read().lower()

        if data is not None:
            self.data = data

        self.block_size = block_size
        self.vocab = sorted(list(set(self.data)))

        self.length = len(self.data)
        self.vocab_size = len(self.vocab)

        self.char_to_index = { c:i for i,c in enumerate(self.vocab)}
        self.index_to_char = { i:c for i,c in enumerate(self.vocab)}

    def __len__(self):
        return self.length-self.block_size

    def __getitem__(self, i):
        seq = self.data[i:i+self.block_size+1]
        seq_ids = [self.char_to_index[c] for c in seq]

        x = torch.tensor(seq_ids[:-1], dtype=torch.long)
        y = torch.tensor(seq_ids[1:], dtype=torch.long)
        return x, y

    def train_test_split(self, train_fraction=0.5):
        Ns = int(train_fraction*self.length)

        data_train = self.data[:Ns]
        data_test = self.data[Ns:]

        dataset_train = CharDataset(data=data_train, block_size=self.block_size)
        dataset_test = CharDataset(data=data_test, block_size=self.block_size)

        dataset_train.vocab = self.vocab
        dataset_train.vocab_size = self.vocab_size
        dataset_train.char_to_index = self.char_to_index
        dataset_train.index_to_char = self.index_to_char

        dataset_test.vocab = self.vocab
        dataset_test.vocab_size = self.vocab_size
        dataset_test.char_to_index = self.char_to_index
        dataset_test.index_to_char = self.index_to_char

        return dataset_train, dataset_test
