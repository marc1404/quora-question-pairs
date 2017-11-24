import pandas as pd

train_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data.csv")
test_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data.csv")

print('Total number of question pairs for training: {}'.format(len(train_df)))
