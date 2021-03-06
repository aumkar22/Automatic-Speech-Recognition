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


def wav2numpy(wav_list: List[str]) -> Tuple[np.ndarray, List[int]]:

    """
    Function to load wav files into numpy arrays

    :param wav_list: List of paths to wav files
    :return: Lists of sampling frequencies and read numpy arrays
    """

    wav_data_list = []
    wav_to_ignore_index = []

    for wav_index, wav_file in enumerate(wav_list):
        sampling_rate, wav_data = wavfile.read(wav_file)
        # Since Google speech dataset was collected with a sampling frequency of 16KHz,
        # data with less than 15k length was ignored.
        if len(wav_data) < 15000:
            wav_to_ignore_index.append(wav_index)
        else:
            wav_data_list.append(wav_data)
    return np.array(wav_data_list), wav_to_ignore_index
