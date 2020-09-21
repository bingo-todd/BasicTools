import numpy as np


def normalize(x, axis):
    x_min = np.expand_dims(np.min(x, axis=axis), axis=axis)
    x_max = np.expand_dims(np.max(x, axis=axis), axis=axis)
    x_norm = (x - x_min)/(x_max-x_min)
    return x_norm 
