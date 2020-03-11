from data.LSTM_data import *
from model.LSTM import *
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words))

def main():
    print('***********************************')
    print('The Feynman Like Sentence Generator')
    print('***********************************')

    print('Load Data')
    # parameters
    seq_len = 32
    batch_size = 32
    learning_rate = 0.001
    epochs = 1
    # data
    data = FeynmanDataset('data/feynman_lectures.csv', seq_len)
    train_loader = DataLoader(data, batch_size=batch_size)

    print('Load Model')
    net = SentenceGenerator(data.n_vocab, seq_len, 128, 128)
    net = net.to(device)

    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        state_h, state_c = net.zero_state(batch_size)
        iteration = 0
        for input, output in train_loader:
            if input.shape != (1,batch_size,128):
                break
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            input = input.to(device)
            output = output.to(device)

            net.train()
            optimizer.zero_grad()
            logits, (state_h, state_c) = net(input, (state_h, state_c))
            loss = criterion(logits.transpose(1,2), output)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            loss_value = loss.item()
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            iteration += 1
            if iteration%100 == 1:
                print("loss: ", loss_value)

    print('Start Prediction')
    predict(device, net, 'I', data.n_vocab, data.vocab_to_int, data.int_to_vocab, top_k=5)


if __name__ == '__main__':
    main()
