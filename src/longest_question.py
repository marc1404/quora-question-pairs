def maximum(questions):
    return questions.map(lambda x: len(x)).max()


def find(train_df, test_df):
    return max(
        maximum(train_df.question1),
        maximum(train_df.question2),
        maximum(test_df.question1),
        maximum(test_df.question2)
    )
