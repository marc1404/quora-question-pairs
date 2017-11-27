import src.question_to_words as question_to_words
import src.words_to_vector as words_to_vector


def use_word2vec():
    words_to_vector.use_word2vec()


def transform(question):
    words = question_to_words.transform(question)
    vector = words_to_vector.transform(words)

    return vector
