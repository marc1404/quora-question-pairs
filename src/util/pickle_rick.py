import pickle


def dump(object, file_path):
    file = open(file_path, 'wb')

    pickle.dump(object, file)
    file.close()


def load(file_path):
    file = open(file_path, 'rb')
    object = pickle.load(file)

    file.close()

    return object
