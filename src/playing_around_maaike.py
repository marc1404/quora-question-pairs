import pandas as pd

train_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data.csv")
test_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data.csv")
trainmerge_df = pd.read_csv(r"C:\Users\Maaik\quora-question-pairs\data\train_data_merge.csv")

#%%
print('Total number of question pairs for training: {}'.format(len(train_df)))

#%%
train_df.head()

#%%
print('Duplicate pairs: {}%'.format(round(trainmerge_df['is_duplicate'].mean()*100, 2)))

#%%
