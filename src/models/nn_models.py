import tensorflow.keras as tf

from abc import ABC, abstractmethod
from typing import List


class NnModel(ABC):

    """
    Generic neural network model class. Any deep learning architecture
    in this project should inherit this class for a consistent interface
    """

    def __init__(self, features: int = 16000, out: int = 35):

        """
        Initialize the model with hyperparameters

        :param features: Preprocessed audio input
        :param out: Number of classes
        """

        self.features = features
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
        model: tf.Model,
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

    # @abstractmethod
    # def model_callbacks(self) -> List[tf.callbacks]:
    #
    #     """
    #     Function for adding model callbacks
    #
    #     :return: List of callbacks
    #     """
    #     pass
