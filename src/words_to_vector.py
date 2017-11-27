from nltk.corpus import stopwords
import src.word2vec as word2vec

stops = set(stopwords.words('english'))
in_vocabulary = lambda x: False

word_to_index = dict()
index_to_word = ['<unknown>']


def use_word2vec():
    global n_vocabulary

    word2vec.load()

    in_vocabulary = word2vec.in_vocabulary


def transform(words):
    vector = []

    for word in words:
        assert_index(word)

        index = word_to_index[word]

        vector.append(index)

    return vector


def is_dispensable(word):
    return word in stops and not in_vocabulary(word)


def assert_index(word):
    if word in word_to_index:
        return

    word_to_index[word] = len(index_to_word)

    index_to_word.append(word)
