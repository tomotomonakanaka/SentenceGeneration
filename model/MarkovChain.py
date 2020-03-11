from nltk.util import ngrams
import pandas as pd
import re

def MakeNgram(sentence,N):
    s = sentence.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    if len(tokens) >= N:
        output = list(ngrams(tokens, N))
        return output

def NgramDictGenerator(ngrams, N):
    NgramDict = {}
    for i in range(len(ngrams)):
        if ngrams[i][:N-1] not in NgramDict:
            NgramDict[ngrams[i][:N-1]] = [ngrams[i][N-1]]
        else:
            NgramDict[ngrams[i][:N-1]].append(ngrams[i][N-1])
    return NgramDict

def MakeNgramDict(path, N):
    ngrams = []
    df = pd.read_csv(path)
    sentences = df['Sentences'].to_numpy()
    for i in range(sentences.shape[0]):
        output = MakeNgram(sentences[i], N)
        if output != None:
            ngrams.extend(output)
    NgramDict = NgramDictGenerator(ngrams, N)
    return NgramDict
