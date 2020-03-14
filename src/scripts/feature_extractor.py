import numpy as np

from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
from typing import List


def mfcc_extractor(wav_array: np.ndarray) -> np.ndarray:

    """
    Function to extract Mel Frequency Cepstral Coefficients (MFCC) features from wav numpy arrays

    :param wav_array: Numpy array of wav files
    :return: Extracted MFCC features
    """

    return mfcc(wav_array, winlen=0.03, numcep=40, nfilt=40)


def data_encode(label: List[str]) -> np.ndarray:

    """
    Function to encode data labels for training

    :param label: List of string labels
    :return: Array of encoded labels
    """

    encoder = LabelEncoder()
    y_encode = encoder.fit_transform(label)

    return encoder.fit_transform(y_encode)
