from keras.preprocessing.sequence import pad_sequences

sequences = [
    [5, 10, 4],
    [0, 20],
    [42, 30, 10, 12, 5]
]

padded = pad_sequences(sequences)

print(padded)
