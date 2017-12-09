# Word embeddings try to "compress" large one-hot word ectors into much smaller vectors
# which preserve some of the meaning and context of the word.
# Word2Vec is the most common process of word embedding.

from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

from nltk.corpus import stopwords
from string import punctuation

import re
import pandas as pd
import numpy as np
import tensorflow as tf
import csv

# def load_data(train_df):
#    with open('data/train_data_for_embed.csv', 'r') as f:
#        reader = csv.reader(f)
#        train_df = list(reader)
#    print(train_df)

#    return (train_df)


train = pd.read_csv('data/train_data_for_embed.csv', header=0)

a = 0
for i in range(a,a+10):
    print(train.Question[i])
    print()

# %%

def text_to_word_list(text, remove_stopwords=True):
    # pre process and convert text into a list of words
    text = str(text)
    text = text.lower().split() # convert words to lower case

    if remove_stopwords:
        stops = set(stopwords.words('english'))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    clean = re.sub(r"[^A-Za-z0-9]", " ", text) # = Remove non-letters
    words = clean.split()

    # Remove punctuation from text
    #text = ''.join([c for c in text if c not in punctuation])
    return(words)

print(stops)

def process_questions(question_list, questions, question_list_name, dataframe):
    # transform questions and display the progress of doing this
    for question in questions:
        question_list.append(text_to_word_list(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress,1)))

train_question = []
process_questions(train_question, train.Question, 'train_Question', train)

a = 0
for i in range(a,a+10):
    print(train_question[i])
    print()

# %%

def build_dataset(words, n_words):
    # Process raw inputs into a dataset
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

print(data)
