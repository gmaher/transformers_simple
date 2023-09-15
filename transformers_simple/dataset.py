import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, filename, block_size):
        self.fn = filename
        self.block_size = block_size

        self.data = open(self.fn,'r').read().lower()
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

        x = torch.tensor(seq_ids[:-1])
        y = torch.tensor(seq_ids[1:])
        return x, y
