import pandas as pd
from ast import literal_eval


def parse(path, columns=[]):
    data = pd.read_csv(path)

    for column in columns:
        data[column] = data[column].apply(literal_eval)

    return data
