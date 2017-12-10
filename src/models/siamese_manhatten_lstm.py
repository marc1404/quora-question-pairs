from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.layers import Input, Embedding, LSTM, Merge
from keras.models import Model
from keras.optimizers import Adadelta
import src.util.csv as csv
import src.longest_question as longest_question
from src.util.create_submission import create_submission
import src.util.pickle_rick as pickle_rick
import matplotlib.pyplot as plt


def zero_padding(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length)


columns = ['question1', 'question2']
train_df = csv.parse('data/train_vector.csv', columns)
test_df = csv.parse('data/test_vector.csv', columns)
train_df = train_df.head(10000)
# %%

embeddings = pickle_rick.load('data/embeddings.pckl')
# %%

longest = longest_question.find(train_df, test_df)

print(longest)
# %%

questions1 = zero_padding(train_df.question1, longest)
questions2 = zero_padding(train_df.question2, longest)

test_questions1 = zero_padding(test_df.question1, longest)
test_questions2 = zero_padding(test_df.question2, longest)

print('Done with zero padding')
# %%

X = {
    'left': questions1,
    'right': questions2
}

X_test = {
    'left': test_questions1,
    'right': test_questions2
}

Y = train_df.is_duplicate.values

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 32
n_epoch = 1
embedding_dim = 300
validation_split = 0.1
early_stopping = EarlyStopping(monitor='val_loss', patience=1, mode='auto')


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


left_input = Input(shape=(longest,), dtype='int32')
right_input = Input(shape=(longest,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=longest, trainable=False)

encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

malstm = Model([left_input, right_input], [malstm_distance])

optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

malstm_trained = malstm.fit([X['left'], X['right']], Y, batch_size=batch_size, epochs=n_epoch, validation_split=validation_split, callbacks=[early_stopping])
# %%

plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# %%

plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
# %%

create_submission(malstm, [X_test['left'], X_test['right']])
# %%
