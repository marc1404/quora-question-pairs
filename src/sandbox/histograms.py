import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('data/train_data.csv')
df_test = pd.read_csv('data/test_data.csv')
df_labels = pd.read_csv('data/train_labels.csv')

df_train = df_train.combine_first(df_labels)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color='green', normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color='red', normed=True, alpha=0.7, label='test')
plt.legend()
plt.xlabel('Characters per question', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.savefig('output/char_per_question.svg')
plt.show()
# %%

print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean() * 100, 2)))
# %%

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color='green', normed=True, label='train')
plt.hist(dist_test, bins=50, range=[0, 50], color='red', normed=True, alpha=0.7, label='test')
plt.legend()
plt.xlabel('Words per question', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.savefig('output/word_per_question.svg')
# %%
