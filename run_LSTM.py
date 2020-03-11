from data.LSTM_data import *
from model.LSTM import *
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unicodetoascii(text):

    TEXT = (text.
    		replace(b'\xe2\x80\x99', b"'").
            replace(b'\xc3\xa9', b'e').
            replace(b'\xe2\x80\x90', b'-').
            replace(b'\xe2\x80\x91', b'-').
            replace(b'\xe2\x80\x92', b'-').
            replace(b'\xe2\x80\x93', b'-').
            replace(b'\xe2\x80\x94', b'-').
            replace(b'\xe2\x80\x94', b'-').
            replace(b'\xe2\x80\x98', b"'").
            replace(b'\xe2\x80\x9b', b"'").
            replace(b'\xe2\x80\x9c', b'"').
            replace(b'\xe2\x80\x9c', b'"').
            replace(b'\xe2\x80\x9d', b'"').
            replace(b'\xe2\x80\x9e', b'"').
            replace(b'\xe2\x80\x9f', b'"').
            replace(b'\xe2\x80\xa6', b'...').
            replace(b'\xe2\x80\xb2', b"'").
            replace(b'\xe2\x80\xb3', b"'").
            replace(b'\xe2\x80\xb4', b"'").
            replace(b'\xe2\x80\xb5', b"'").
            replace(b'\xe2\x80\xb6', b"'").
            replace(b'\xe2\x80\xb7', b"'").
            replace(b'\xe2\x81\xba', b"+").
            replace(b'\xe2\x81\xbb', b"-").
            replace(b'\xe2\x81\xbc', b"=").
            replace(b'\xe2\x81\xbd', b"(").
            replace(b'\xe2\x81\xbe', b")")

                 )
    return TEXT

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

    sen = u' '.join(words)
    sen = sen.encode('utf-8')
    sen = unicodetoascii(sen)
    print(sen)

def main():
    print('***********************************')
    print('The Feynman Like Sentence Generator')
    print('***********************************')

    print('Load Data')
    # parameters
    seq_len = 128
    batch_size = 1
    learning_rate = 0.001
    epochs = 10
    train=False
    modelPath = 'model/lstm'
    # data
    data = FeynmanDataset('data/feynman_lectures.csv', seq_len)
    train_loader = DataLoader(data, batch_size=batch_size)

    print('Load Model')
    net = SentenceGenerator(data.n_vocab, seq_len, 128, 128)
    net = net.to(device)

    # train
    if train==True:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(epochs):
            state_h, state_c = net.zero_state(batch_size)
            iteration = 0
            for input_tensor, output in train_loader:
                state_h = state_h.to(device)
                state_c = state_c.to(device)
                input_tensor = input_tensor.to(device)
                output = output.to(device)

                net.train()
                optimizer.zero_grad()
                logits, (state_h, state_c) = net(input_tensor, (state_h, state_c))
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
        torch.save(net, modelPath)

    else:
        net = torch.load(modelPath)


    print('Start Prediction')
    for i in range(10):
        while(True):
            word = input('Please Specify The First Word: ')
            if word in data.vocab_to_int:
                break
            else:
                print('Sorry, this word does not exist, please type another word.')
        print('*******Generated Sentence is*******')
        predict(device, net, [word], data.n_vocab, data.vocab_to_int, data.int_to_vocab, top_k=5)
        print('***********************************')


if __name__ == '__main__':
    main()
