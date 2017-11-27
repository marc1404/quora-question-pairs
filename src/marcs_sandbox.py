import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import src.train_data as train_data

train_df = train_data.load()
test_df = pd.read_csv('data/train_data.csv')

train_df.head()
# %%

test_df.head()
# %%


def text_to_words(text):
    text = str(text)
    text = text.lower()
    return text.split()


word_to_index = dict()
index_to_word = ['<unknown>']
question_columns = ['question1', 'question2']
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
stops = set(stopwords.words('english'))

print('Done loading word2vec')
# %%

for data in [train_df, test_df]:
    for index, row in data.iterrows():
        for question_column in question_columns:
            question = row[question_column]
            words = text_to_words(question)
            question_as_vector = []

            for word in words:
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in word_to_index:
                    word_to_index[word] = len(index_to_word)
                    index_to_word.append(word)

                question_as_vector.append(word_to_index[word])

            data.set_value(index, question_column, question_as_vector)

train_df.head()
# %%

test_df.head()
# %%
