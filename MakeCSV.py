import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import sent_tokenize

def get_feynman_lectures_data():
    Sentences = []
    # get Feynman Lecture 1
    for i in range(1, 53):
        if i < 10:
            page = requests.get('https://www.feynmanlectures.caltech.edu/I_0{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        else:
            page = requests.get('https://www.feynmanlectures.caltech.edu/I_{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        p_list = soup.find_all('p')
        for j in range(6,len(p_list)):
            Sentences.append(p_list[j].get_text().replace('\r\n',' ').replace('                         ', ' '))

    # get Feynman Lecture 2
    for i in range(1, 43):
        if i < 10:
            page = requests.get('https://www.feynmanlectures.caltech.edu/II_0{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        else:
            page = requests.get('https://www.feynmanlectures.caltech.edu/II_{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        p_list = soup.find_all('p')
        for j in range(6,len(p_list)):
            Sentences.append(p_list[j].get_text().replace('\r\n',' ').replace('                         ', ' '))

    # get Feynman Lecture 3
    for i in range(1, 22):
        if i < 10:
            page = requests.get('https://www.feynmanlectures.caltech.edu/III_0{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        else:
            page = requests.get('https://www.feynmanlectures.caltech.edu/III_{}.html'.format(i))
            soup = BeautifulSoup(page.content, 'html.parser')
        p_list = soup.find_all('p')
        for j in range(6,len(p_list)):
            Sentences.append(p_list[j].get_text().replace('\r\n',' ').replace('                         ', ' '))


    NewSentences = []

    # delete math equations
    for i in range(len(Sentences)):
        for j in sent_tokenize(Sentences[i]):
            if '$' in j or '\\' in j:
                continue
            else:
                NewSentences.append(j)

    df = pd.DataFrame(columns=['Sentences'])
    df['Sentences'] = NewSentences

    # save data
    df.to_csv('data/feynman_lectures.csv')


if __name__ == '__main__':
    get_feynman_lectures_data()
