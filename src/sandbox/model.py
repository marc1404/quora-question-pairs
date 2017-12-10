from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import src.util.csv as csv
import src.longest_question as longest_question
from src.util.create_submission import create_submission


def zero_padding(sequences, max_length):
    return pad_sequences(sequences, maxlen=max_length, padding='post')


columns = ['question1', 'question2']
train_df = csv.parse('data/train_vector.csv', columns)
test_df = csv.parse('data/test_vector.csv', columns)

print(len(train_df))
# %%

longest = longest_question.find(train_df, test_df)

questions1 = zero_padding(train_df.question1, longest)
questions2 = zero_padding(train_df.question2, longest)

test_questions1 = zero_padding(test_df.question1, longest)
test_questions2 = zero_padding(test_df.question2, longest)

print(longest)
# %%

X = np.concatenate((questions1, questions2), axis=1)
y = train_df.is_duplicate

print(X.shape)
# %%

X_predict = np.concatenate((test_questions1, test_questions2), axis=1)

print(X_predict.shape)
# %%

# X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)
# %%

model = Sequential()

model.add(Dense(64, input_dim=longest * 2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit(X, y, epochs=3, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss')])
# %%

# evaluation = model.evaluate(X_validation, y_validation, batch_size=32)

# print(evaluation)
# %%

create_submission(model, X_predict)
# %%
