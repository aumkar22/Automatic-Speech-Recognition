import pandas as pd
import numpy as np

from typing import List, Tuple
from scipy.io import wavfile

from src.util.definitions import *

# Loading test and validation file names in pandas dataframe
test_files = pd.read_csv(
    str(DATA_FOLDER + "/" + "data_info" + "/" + "testing_list.txt"), sep=" ", header=None
)[0].tolist()
val_files = pd.read_csv(
    str(DATA_FOLDER + "/" + "data_info" + "/" + "validation_list.txt"), sep=" ", header=None
)[0].tolist()


def get_test_val_labels_list(dataset: pd.DataFrame) -> Tuple[List[str], List[str]]:

    """
    Function to return test or validation file paths and labels

    :param dataset: Test or validation dataframe
    :return: Lists of test or validation file paths and corresponding labels
    """

    label = [os.path.dirname(i) for i in dataset]
    data_files = [os.path.join(DATA_PATH, f) for f in dataset if f.endswith(".wav")]

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

    data_list = []
    for root, dirs, files in os.walk(DATA_PATH):
        data_list += [root + "/" + f for f in files if f.endswith(".wav")]

    train_list = list(set(data_list) - set(test_list) - set(val_list))
    train_lab = [os.path.basename(os.path.dirname(i)) for i in train_list]

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
