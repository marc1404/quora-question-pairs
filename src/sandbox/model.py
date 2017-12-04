from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import src.util.csv as csv
import src.longest_question as longest_question
import pandas as pd


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

model = Sequential()

model.add(Dense(64, input_dim=longest * 2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)
# %%

predictions = model.predict(X_predict, verbose=1)

print(predictions)
# %%

rounded = [int(round(x[0])) for x in predictions]

print(rounded)
# %%

submission_df = pd.DataFrame(index=test_df.test_id, columns=['is_duplicate'], dtype=np.uint)
submission_df.index.name = 'test_id'
submission_df.is_duplicate = rounded

submission_df.to_csv('data/submission.csv')
# %%
