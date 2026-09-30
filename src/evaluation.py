import numpy as np


def sum_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    sse = sum_squared_error(y_true, y_pred)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (sse / sst)