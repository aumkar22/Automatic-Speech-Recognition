import numpy as np

from typing import Tuple, List
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from python_speech_features import mfcc


def data_length_fix(signal: np.ndarray, number_samples: int = 16000) -> np.ndarray:

    """
    Since the Google Speech Data was collected at sampling rate of 16KHz, the total number
    of samples per data signal should be 16000. If the total number of samples are more than
    16000, the extra samples are excluded. If the total number of samples are less than
    16000, then zero padding is done.

    :param signal: Speech signal
    :param number_samples: Number of samples (default is 16000 as sampling frequency is 16KHz)
    :return: Speech signal with fixed length of 16000
    """

    signal_list = list(signal)
    if len(signal_list) >= number_samples:
        signal_list = signal_list[:number_samples]  # Excluding extra samples
    else:
        signal_list.extend([0] * (number_samples - len(signal_list)))  # Zero padding

    return np.array(signal_list)


def fit_standardize(train_data: np.ndarray) -> StandardScaler:

    """
    Function to standardize the data. Returns a scaler object which stores mean and variance of
    the training data. Can be applied to test / validation data for evaluation

    :param train_data: Array of training data
    :return: Scaler object storing mean and variance of the training data
    """

    scaler = StandardScaler()
    scaler.fit(train_data)

    return scaler


def apply_standardize(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:

    """
    Function to apply standardizing to the test / validation data

    :param data: Data for standardization
    :param scaler: Scaler object which has stored training mean and variance
    :return: Standardized data
    """

    standardized_data = scaler.transform(data)

    return standardized_data


def data_balancing(
    train_data: np.ndarray, train_labels: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Balancing the training data by over sampling every class except the majority class

    :param train_data: Training data
    :param train_labels: Training data labels
    :param seed: Seed for reproducibility
    :return: Balanced training data and corresponding labels
    """

    resampler = RandomOverSampler(random_state=seed)
    train_data_resampled, train_labels_resampled = resampler.fit_resample(train_data, train_labels)

    return train_data_resampled, train_labels_resampled


def data_encode(label: List[str]) -> np.ndarray:

    """
    Function to encode data labels for training

    :param label: List of string labels
    :return: Array of encoded labels
    """

    encoder = LabelEncoder()

    return encoder.fit_transform(label)


def mfcc_extractor(wav_array: np.ndarray) -> np.ndarray:

    """
    Function to extract Mel Frequency Cepstral Coefficients (MFCC) features from wav numpy arrays
    :param wav_array: Numpy array of wav files
    :return: Extracted MFCC features
    """

    return mfcc(wav_array, winlen=0.03, numcep=40, nfilt=40).T
