import pandas as pd
import numpy as np
import src.util.model_image as model_image

test_df = pd.read_csv('data/test_data.csv')


def create_submission(model, X, path='output/submission.csv'):
    predictions = model.predict(X, verbose=1)
    rounded = [int(round(x[0])) for x in predictions]
    submission_df = pd.DataFrame(index=test_df.test_id, columns=['is_duplicate'], dtype=np.uint)
    submission_df.index.name = 'test_id'
    submission_df.is_duplicate = rounded

    submission_df.to_csv(path)
    model_image.save(model)
