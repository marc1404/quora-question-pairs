import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

train_df = pd.read_csv('data/train_vector.csv')
test_df = pd.read_csv('data/test_vector.csv')

# time for preparing and validation of the data
max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

print(max_seq_length)
