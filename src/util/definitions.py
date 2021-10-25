import pandas as pd
from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent.parent

DATA_FOLDER: Path = PROJECT_PATH / "data"
DATA_PATH: Path = DATA_FOLDER / "speech_data"
FEATURES_PATH: Path = DATA_FOLDER / "features"
DATA_INFO: Path = DATA_FOLDER / "data_info"
MODEL_PATH: Path = DATA_FOLDER / "models"
MODEL_SCRIPTS_PATH = PROJECT_PATH / "src" / "models"
TENSORBOARD_PATH: Path = DATA_FOLDER / "logs"
CACHE_PATH: Path = DATA_FOLDER / "logs" / "cache"
CONFIG_PATH: Path = PROJECT_PATH / "src" / "util" / "config"
PREPROCESSED_PATH: Path = DATA_FOLDER / "preprocessed"
METRICS_PATH: Path = DATA_FOLDER / "metrics"
PLOTS_PATH: Path = PROJECT_PATH / "plots"
TEST_FILE: Path = DATA_INFO / "testing_list.txt"
VALIDATION_FILE: Path = DATA_INFO / "validation_list.txt"

# Loading test and validation file names in pandas dataframe
test_list_from_df = pd.read_csv(TEST_FILE, sep=" ", header=None)[0].tolist()
val_list_from_df = pd.read_csv(VALIDATION_FILE, sep=" ", header=None)[0].tolist()


classes = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]
