import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

# load training and test data sets
train_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data.csv")
test_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\test_data.csv")
embedding_file = (r"C:\Users\Maaik\quora-question-pairs\GoogleNews-vectors-negative300.bin.gz")

stops = set(stopwords.words('english'))


def text_to_word_list(text):
    # pre process and convert text into a list of words
    text = str(text)
    text = text.lower()
    text = text.split()
    return text

# this is going to be our vocabulary, we will give it a word and it will give us the index
word_to_index = dict()
index_to_word = []
question_columns = ['question1', 'question2']

inverse_vocabulary = ['<unk>']
# '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
# because if you substract 0 from anything it will stay the same and there is no use for that later

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

            # replaces questions with lists of word indices:
            dataset.set_value(index, question_column, question_as_vector)


# try out pandas and see if I can get the first 5 from both

# %%
train_df.head()
# %%
test_df.head()
