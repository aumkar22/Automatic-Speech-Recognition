import pandas as pd
from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent.parent

DATA_FOLDER: Path = PROJECT_PATH / "data"
DATA_PATH: Path = DATA_FOLDER / "speech_data"
FEATURES_PATH: Path = DATA_FOLDER / "features"
DATA_INFO: Path = DATA_FOLDER / "data_info"
MODEL_PATH: Path = DATA_FOLDER / "models"
TESNORBOARD_PATH: Path = DATA_FOLDER / "logs"

TEST_FILE: Path = DATA_INFO / "testing_list.txt"
VALIDATION_FILE: Path = DATA_INFO / "validation_list.txt"

# Loading test and validation file names in pandas dataframe
test_df = pd.read_csv(TEST_FILE, sep=" ", header=None)[0].tolist()
val_df = pd.read_csv(VALIDATION_FILE, sep=" ", header=None)[0].tolist()
