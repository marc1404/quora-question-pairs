from gensim.models import KeyedVectors
import src.timer as timer

vectors_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = None


def load():
    global word2vec

    timer.start('word2vec')

    word2vec = KeyedVectors.load_word2vec_format(vectors_path, binary=True)

    timer.end('word2vec')


def is_in_vocabulary(word):
    return word in word2vec.vocab
