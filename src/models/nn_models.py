import keras

from abc import ABC, abstractmethod
from typing import List, Generator


class NnModel(ABC):

    """
    Generic neural network model class. Any deep learning architecture
    in this project should inherit this class for a consistent interface
    """

    def __init__(self, features: int, channels: int, out: int, batchsize: int):

        """
        Initialize the model with hyperparameters

        :param features: Features dimension (Time)
        :param channels: MFCC coefficients
        :param out: Number of classes
        :param batchsize: Training batch size
        """

        self.features = features
        self.channels = channels
        self.out = out
        self.batchsize = batchsize

    @abstractmethod
    def model_architecture(self) -> keras.Model:

        """"
        Function to build model architecture
        """
        pass

    @abstractmethod
    def model_compile(self) -> keras.Model:

        """
        Function to compile the model

        :return: Returns a compiled model
        """
        pass

    @abstractmethod
    def batch_generator(self) -> Generator:

        """
        Function for batch generator

        :return: Yields batches of data and labels
        """
        pass

    @abstractmethod
    def step_decay(self, epoch: int) -> float:

        """
        Step decay function for learning rate scheduling during training

        :param epoch: Epoch number for step decay
        :return: Decayed learning rate
        """
        pass

    @abstractmethod
    def model_callbacks(self) -> List[keras.callbacks]:

        """
        Function for adding model callbacks

        :return: List of callbacks
        """
        pass
