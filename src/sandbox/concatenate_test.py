import numpy as np

questions1 = [
    [1, 2, 3],
    [4, 5, 6]
]
questions2 = [
    [1, 2, 3],
    [4, 5, 6]
]

result = np.concatenate((questions1, questions2), axis=1)

print(result)
