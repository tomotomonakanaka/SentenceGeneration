import torch
import numpy as np
from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader

class FeynmanDataset(Dataset):
    def __init__(self, data_path, seq_len):
        df = pd.read_csv(data_path)
        Sentences = df['Sentences'].to_numpy()
        text = []
        for i in range(len(Sentences)):
            text.extend(word_tokenize(Sentences[i]))
        word_counts = Counter(text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {k: w for k,w in enumerate(sorted_vocab)}
        self.vocab_to_int = {w: k for k,w in self.int_to_vocab.items()}
        self.n_vocab = len(self.int_to_vocab)
        self.int_text = [self.vocab_to_int[w] for w in text]
        self.out_text = self.int_text[1:]
        self.out_text.append(self.int_text[0])

        self.input = []
        self.output = []
        for i in range(int(len(self.int_text)/seq_len)):
            self.input.append(self.int_text[i*seq_len: (i+1)*seq_len])
            self.output.append(self.out_text[i*seq_len: (i+1)*seq_len])
        self.input = torch.LongTensor(self.input)
        self.output = torch.LongTensor(self.output)

    def __getitem__(self, i):
        return self.input[i], self.output[i]

    def __len__(self):
        return len(self.input)
