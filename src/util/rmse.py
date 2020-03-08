import keras.backend as k


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the root mean squared error.
    :param y_true: The ground truth values.
    :param y_pred: The predicted values.
    :return: The RMSE based on the ground and predicted values.
    """

    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))
