import torch
import torch.nn as nn

class SentenceGenerator(nn.Module):
    def __init__(self, n_vocab, seq_len, embedding_size, lstm_size):
        super(SentenceGenerator, self).__init__()
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.fc = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        feature = self.embedding(x)
        output, state = self.lstm(feature, prev_state)
        logits = self.fc(output)
        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size), torch.zeros(1, batch_size, self.lstm_size))
