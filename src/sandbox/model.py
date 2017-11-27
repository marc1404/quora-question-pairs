import src.util.csv as csv
import src.longest_question as longest_question

columns = ['question1', 'question2']
train_df = csv.parse('data/train_vector.csv', columns)
test_df = csv.parse('data/test_vector.csv', columns)

print(len(train_df))
# %%

longest = longest_question.find(train_df, test_df)

print(longest)
