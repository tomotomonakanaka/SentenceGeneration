from model.MarkovChain import *
import numpy as np

def main():
    print('***********************************')
    print('The Feynman Like Sentence Generator')
    print('***********************************')
    print('Load Model')
    ThreeGram = MakeNgramDict('feynman_lectures.csv', 3)
    TwoGram = MakeNgramDict('feynman_lectures.csv', 2)
    for i in range(10):
        while(True):
            word = input('Please Specify The First Word: ')
            if (word,) in TwoGram:
                break
            else:
                print('Sorry, this word does not exist, please type another word.')
        sen = ''
        pre = (word, np.random.choice(TwoGram[(word,)]))
        sen += word + ' ' + pre[-1]
        while(True):
            if pre not in ThreeGram:
                break
            pre = (pre[-1], np.random.choice(ThreeGram[pre]))
            sen += ' ' + pre[-1]
        print('*******Generated Sentence is*******')
        print(sen+'.')
        print('***********************************')


if __name__ == '__main__':
    main()
