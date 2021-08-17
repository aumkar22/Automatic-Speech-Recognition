import tensorflow.keras as tf

from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)


class NnModel(ABC):

    """
    Generic neural network model class. Any deep learning architecture
    in this project should inherit this class for a consistent interface
    """

    def __init__(self, num_ceps: int = 40, num_frames: int = 98, out: int = 35):

        """
        Initialize the model with hyperparameters

        :param num_ceps: Preprocessed audio input
        :param out: Number of classes
        """

        self.num_ceps = num_ceps
        self.num_frames = num_frames
        self.input_shape = (self.num_ceps, self.num_frames, 1)
        self.out = out

    @abstractmethod
    def model_architecture(self) -> tf.Model:

        """"
        Function to build model architecture
        """
        pass

    @abstractmethod
    def model_compile(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> tf.Model:

        """
        Function to compile the model

        :return: Returns a compiled model
        """
        pass

    @staticmethod
    def _step_decay(epoch, learning_rate) -> float:

        """
        Drop learning rate every 10 epochs

        :param epoch: Epoch index
        :param learning_rate: Learning rate at every epoch index
        :return: Dropped learning rate
        """

        drop = 0.4
        epochs_drop = 10.0
        new_lr = learning_rate * (drop ** ((1 + epoch) // epochs_drop))

        if new_lr < 4e-5:
            new_lr = 4e-5

        print(f"Changing learning rate to {new_lr}")

        return new_lr

    @classmethod
    def model_callbacks(cls, save_path: Path, log_path: Path) -> List:

        """
        Function for adding model callbacks

        :param save_path: Model checkpoint save path
        :param log_path: Tensorboard log path
        :return: List of callbacks
        """

        if not save_path.exists():
            save_path.mkdir(exist_ok=True, parents=True)
        if not log_path.exists():
            log_path.mkdir(exist_ok=True, parents=True)

        tensorboard_callback = TensorBoard(log_dir=log_path)
        step_decay_lr = LearningRateScheduler(cls._step_decay)
        model_checkpoint = ModelCheckpoint(
            filepath=str(save_path), monitor="val_sparse_categorical_accuracy", save_best_only=True
        )
        early_stopper = EarlyStopping(
            monitor="val_sparse_categorical_accuracy", patience=10, verbose=1
        )

        return [tensorboard_callback, step_decay_lr, model_checkpoint, early_stopper]
