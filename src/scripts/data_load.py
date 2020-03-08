import numpy as np
import pandas as pd

from scipy.io import wavfile
from python_speech_features import mfcc

from src.definitions import *

test_files = pd.read_csv((DATA_PATH / "data_info" / "testing_list.txt"), sep=" ", header=None)[
    0
].tolist()
val_files = pd.read_csv((DATA_PATH / "data_info" / "validation_list.txt"), sep=" ", header=None)[
    0
].tolist()
breakpoint()
