import src.util.csv as csv

columns = ['question1', 'question2']
train_df = csv.parse('data/train_vector.csv', columns)
test_df = csv.parse('data/test_vector.csv', columns)

longest = max(
    train_df.question1.map(lambda x: len(x)).max(),
    train_df.question2.map(lambda x: len(x)).max(),
    test_df.question1.map(lambda x: len(x)).max(),
    test_df.question2.map(lambda x: len(x)).max()
)

print(longest)
