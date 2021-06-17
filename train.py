import yaml
import tensorflow.keras as tf
import argparse
import importlib, inspect

from typing import Dict, Optional, Tuple

from src.models.cnn1_99k import *
from src.scripts.augmenter import *
from src.scripts.generator import BatchGenerator
from src.util.definitions import *
from src.scripts.save_processed_data import save_data
from src.models.eval import EvalVisualize


def train(
    model: NnModel,
    train_data: Tuple[np.ndarray, ...],
    val_data: Tuple[np.ndarray, ...],
    test_data: Tuple[np.ndarray, ...],
    augmentations: Optional[List[Augmentation]],
):

    train_generator = BatchGenerator(train_data[0], train_data[1], augmentations=augmentations)
    validation_generator = BatchGenerator(val_data[0], val_data[1], augmentations=None)
    test_generator = BatchGenerator(test_data[0], test_data[1], augmentations=None)

    compiled_model = model.model_compile()
    callbacks = model.model_callbacks(MODEL_PATH, TENSORBOARD_PATH)

    compiled_model.fit_generator(
        train_generator,
        validation_generator=validation_generator,
        epochs=500,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=6,
    )

    test_predict = compiled_model.predict_generator(
        test_generator, use_multiprocessing=True, workers=6
    )
    predictions = np.argmax(test_predict, 1)

    evaluation_visualization = EvalVisualize(test_data[1], predictions)
    evaluation_visualization.get_metrics(Path(METRICS_PATH / "metrics.pkl"), print_report=True)
    evaluation_visualization.get_confusion_matrix(Path(PLOTS_PATH / "plot.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Select model to train",
        choices=[
            "cnn1_99k",
            "cnn1_100k",
            "cnn1_134k",
            "cnn2_248k",
            "cnn_bilstm_163k",
            "resnet_334k",
        ],
        default="cnn1_99k",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        help="Boolean option to select of data is " "preprocessed",
        choices=["False", "True"],
        default="False",
    )
    args = parser.parse_args()
    model_ = args.model

    model_choose = model_.split("_")[-1]

    for path_ in list(CONFIG_PATH.glob(r"*.yaml")):
        if path_.stem.split("_")[-1] == model_choose:
            params = yaml.safe_load(path_.open())
            break

    for path_ in list(MODEL_SCRIPTS_PATH.glob(r"*.py")):
        if path_.stem.split("_")[-1] == model_choose:
            module = importlib.import_module(f"src.models.{model_}")
            for name, class_ in inspect.getmembers(module, inspect.isclass):
                if str(name)[-1] == "k":
                    module_source = importlib.import_module(f"src.models.{model_}")
                    model_to_train = getattr(module_source, f"{name}")
                    break

    preprocess = args.preprocess

    if preprocess == "False":
        save_data(FEATURES_PATH)

    xtrain = np.load(str(FEATURES_PATH / "train.npy"))
    xval = np.load(str(FEATURES_PATH / "val.npy"))
    xtest = np.load(str(FEATURES_PATH / "test.npy"))
    ytrain = np.load(str(FEATURES_PATH / "train_labels.npy"))
    yval = np.load(str(FEATURES_PATH / "val_labels.npy"))
    ytest = np.load(str(FEATURES_PATH / "test_labels.npy"))
