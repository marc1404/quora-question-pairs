import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

import src.util.csv as csv

question_columns = ['question1', 'question2']

train_df = csv.parse('data/train_vector.csv', question_columns)
test_df = csv.parse('data/test_vector.csv', question_columns)

# time for preparing and validation of the data

# let's find out the max sequence lenght = longest question
def max_sent_length():
    max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

    return max_seq_length

print('max_seq_length: {}' .format(max_sent_length()))

# Need to split the data to 'left' and 'right' inputs (one for each side of the Manhattan Long Short Term Memory network)
validation_size = 40000
training_size = len(train_df) - validation_size
print('training size: {}' .format(training_size))


X = train_df[question_columns]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# split in train validation and test questions 1
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}

# convert labels to their numpy representation
Y_train = Y_train.values
Y_validation = Y_validation.values

# add zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)
