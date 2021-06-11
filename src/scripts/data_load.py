import numpy as np

from typing import List, Tuple
from scipy.io import wavfile

from src.util.definitions import *


def get_test_val_labels_list(dataset: List[str]) -> Tuple[List[str], List[str]]:

    """
    Function to return test or validation file paths and labels

    :param dataset: List of test or validation data wav
    :return: Lists of test or validation file paths and corresponding labels
    """

    label = [str(Path(i).parent) for i in dataset]
    data_files = [str(Path(DATA_PATH / f)) for f in dataset if f.endswith(".wav")]

    return data_files, label


def train_data_labels_list(
    test_list: List[str], val_list: List[str]
) -> Tuple[List[str], List[str]]:

    """
    Function to get training file paths and labels

    :param test_list: List of test file paths
    :param val_list: List of validation file paths
    :return: Lists of training file paths and corresponding labels
    """

    data_list = [str(i) for i in DATA_PATH.glob(r"**/*.wav")]

    train_list = list(set(data_list) - set(test_list) - set(val_list))
    train_lab = [str(Path(i).parent.stem) for i in train_list]

    return train_list, train_lab


def wav2numpy(wav_list: List[str]) -> Tuple[List[int], List[np.ndarray]]:

    """
    Function to load wav files into numpy arrays

    :param wav_list: List of paths to wav files
    :return: Lists of sampling frequencies and read numpy arrays
    """

    sampling_rate_list, wav_data_list = [], []

    for wav_file in wav_list:
        sampling_rate, wav_data = wavfile.read(wav_file)

        sampling_rate_list.append(sampling_rate)
        wav_data_list.append(wav_data)

    return sampling_rate_list, wav_data_list
