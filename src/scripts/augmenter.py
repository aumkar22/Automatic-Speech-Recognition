import numpy as np


def scaling(signal: np.ndarray, scale_factor: float = 0.1, mean: float = 1.0) -> np.ndarray:

    """
    Function to perform scaling augmentation

    :param signal: Input original signal
    :param scale_factor: Standard deviation of the noise
    :param mean: Mean of the noise
    :return: Scaled signal array
    """

    signal_reshaped = np.reshape(signal, (signal.shape[0], 1))

    noise = np.random.normal(loc=mean, scale=scale_factor, size=(1, signal_reshaped.shape[1]))
    scaling_noise = np.matmul(np.ones((signal_reshaped.shape[0], 1)), noise)

    signal_scaled = signal_reshaped * scaling_noise

    return np.array([i for i in np.reshape(signal_scaled, (signal_scaled.shape[0]))])


def jitter(signal: np.ndarray, sigma: float = 0.05, mean: float = 0.0) -> np.ndarray:

    """
    Function to perform jitter augmentation

    :param signal: Input original signal
    :param sigma: Standard deviation of the noise
    :param mean: Mean of the noise
    :return: Signal with additive noise
    """

    noise = np.random.normal(loc=mean, scale=sigma, size=signal.shape)

    return np.array([i for i in (signal + noise)])


# Function taken from https://github.com/PJansson/speech/blob/master/utils/data.py
def timeshift(signal: np.ndarray, max_shift: float = 0.2) -> np.ndarray:

    """
    Function to shift audio in time.

    :param signal: Input original signal
    :param max_shift: Shift bound
    :return: Time shifted signal
    """

    shift = np.random.uniform(-max_shift, max_shift)
    shift = int(len(signal) * shift)
    if shift > 0:
        padded = np.pad(signal, (shift, 0), "constant")
        return np.array(padded[: len(signal)])
    else:
        padded = np.pad(signal, (0, -shift), "constant")
        return np.array(padded[-len(signal) :])
