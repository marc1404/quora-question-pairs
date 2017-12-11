import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('data/test_data.csv')
df_test = pd.read_csv('data/train_data.csv')

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color='#aa2200', normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color='blue', normed=True, alpha=0.7, label='test')
plt.legend()
plt.xlabel('Characters per question', fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.show()
# %%
