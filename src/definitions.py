import os
from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent

DATA_FOLDER = os.path.join(str(PROJECT_PATH), "data")
DATA_PATH = os.path.join(DATA_FOLDER, "speech_data")
FEATURES_PATH = os.path.join(DATA_FOLDER, "features")
