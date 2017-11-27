import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np
from keras.layers import Embedding
import os


# load training and test data sets
train_df = pd.read_csv('data/train_data_merge.csv')
test_df = pd.read_csv('data/test_data.csv')
embedding_file = ('GoogleNews-vectors-negative300.bin.gz')


stops = set(stopwords.words('english'))


def text_to_word_list(text):
    # pre process and convert text into a list of words
    text = str(text)
    text = text.lower()


    # Let's start scrubbing the text ;D and do a little bit of stemming
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text) # removes 's from words
    text = re.sub(r"\'ve", " have ", text) # changes the ending of words 've into: <blank space> have
    text = re.sub (r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text) # remove commas
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"''", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text) < dont know what it does (yet)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" United States ", " america ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r" the us ", " america ", text)


    text = text.split() # split into words by white space
    return text

# this is going to be our vocabulary, we will give it a word and it will give us the index
word_to_index = dict()
index_to_word = ['<unk>']
question_columns = ['question1', 'question2']


#embedding
word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
# %%
print('Done loading word2vec')
# iterate over the questions only of both training and test dataset
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # iterate through the text of both questions of the row
        for question_column in question_columns:
            question = row[question_column]
            words = text_to_word_list(question)
            question_as_vector = [] # question_as_vector = q2n = question number representation

            for word in words:
                if word in stops and word not in word2vec.vocab: #check for unwanted words
                    continue

                if word not in word_to_index:
                    word_to_index[word] = len(index_to_word)
                    index_to_word.append(word)

                question_as_vector.append(word_to_index[word])

            # replaces questions with lists of word indices = number representation:
            dataset.set_value(index, question_column, question_as_vector)

# %%
embedding_dim = 300
embeddings = 1 * np.random.randn(len(word_to_index) + 1, embedding_dim) # this will be the embedding matrix
embeddings[0] = 0 # so that the padding will be ignored

# build the embedding matrix
for word, index in word_to_index.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

# os.remove('data/train_vector.csv')
# os.remove('data/test_vector.csv')

train_df.to_csv('data/train_vector.csv')
test_df.to_csv('data/test_vector.csv')
