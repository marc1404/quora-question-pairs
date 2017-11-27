import pandas as pd
import src.train_data as train_data
import src.question_to_vector as question_to_vector
import src.util.timer as timer

train_df = train_data.load()
test_df = pd.read_csv('data/test_data.csv')

train_df.head()
# %%

test_df.head()
# %%

question_columns = ['question1', 'question2']

# question_to_vector.use_word2vec()
# %%

total_rows = train_df.shape[0] + test_df.shape[0]
current_row = 0
percent = 0

timer.start('vectorize')
print('0%', end='\r')

for data in [train_df, test_df]:
    for index, row in data.iterrows():
        for question_column in question_columns:
            question = row[question_column]
            vector = question_to_vector.transform(question)

            data.set_value(index, question_column, vector)

        current_row += 1
        progress = current_row / total_rows
        new_percent = (int)(progress * 100)

        if new_percent > percent:
            print(str(new_percent) + '%', end='\r')

        percent = new_percent

timer.end('vectorize')
# %%

train_df.head()
# %%

test_df.head()
# %%
