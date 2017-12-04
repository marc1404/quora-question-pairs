from keras.utils import plot_model
import time


def save(model):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    to_file = 'output/model_' + timestamp + '.png'
    plot_model(model, to_file=to_file, show_shapes=True)
