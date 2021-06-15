import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List


class Augmentation(ABC):
    """
    Base class for data augmentations. Other augmentations should extend this class to make sure
    that the interface for augmentations is consistent.
    """

    @abstractmethod
    def augment(self, features: np.ndarray) -> np.ndarray:
        """
        Augment the speech data

        :param features: Data that should be augmented.
        :return: Augmented data
        """
        pass


class JitterAugmentation(Augmentation):
    def __init__(self, mean=0.0, std=0.05):
        """
        Initialize the object using the mean and std which will be used to pull random numbers from
        a normal distribution.

        :param mean:
        :param std:
        """
        self._mean = mean
        self._std = std

    def augment(self, features: np.ndarray) -> np.ndarray:
        """
        Apply noise to the input data based on the mean and standard deviation used to initialize
        this augmentation.

        :param features:
        :return:
        """
        return features + np.random.normal(loc=self._mean, scale=self._std, size=features.shape)


class ScalingAugmentation(Augmentation):
    def __init__(self, mean=1.0, std=0.1):
        """
        Initialize the object using the mean and std which will be used to pull random numbers from
        a normal distribution.

        :param mean:
        :param std:
        """
        self._mean = mean
        self._std = std

    def augment(self, features: np.ndarray) -> np.ndarray:
        """
        Apply a small change in scale to the provided data based on the mean and standard deviation
        that were used to initialize this class.

        :param features:
        :return:
        """
        scaling_factor = np.random.normal(
            loc=self._mean, scale=self._std, size=(1, features.shape[1])
        )
        return features * scaling_factor


class TimeShiftAugmentation(Augmentation):
    def __init__(self, max_shift: float = 0.2):
        """
        Initialize object to shift audio in time

        :param max_shift:
        """

        self.max_shift = max_shift

    # Function taken from https://github.com/PJansson/speech/blob/master/utils/data.py
    def augment(self, features: np.ndarray) -> np.ndarray:

        """

        :param features:
        :return:
        """
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        shift = int(len(features) * shift)
        if shift > 0:
            padded = np.pad(features, (shift, 0), "constant")
            return np.array(padded[: len(features)])
        else:
            padded = np.pad(features, (0, -shift), "constant")
            return np.array(padded[-len(features) :])


class NoChangeAugmentation(Augmentation):
    def augment(self, features: np.ndarray) -> np.ndarray:
        """
        Return the input data without any change. The idea behind this is that you can create a
        list of possible augmentations, while also being able to not augment data.

        :param features:
        :return:
        """
        return features


def apply_augmentations(data: np.ndarray, augmentations: List[Augmentation]) -> np.ndarray:
    """
    Iterate over experiments in the data and per experiment apply a random augmentation from the
    augmentations provided.

    :param data: Data in the shape of (experiments, samples) that will be augmented.
    :param augmentations: List of augmentations from which one will be picked to apply per
                          experiment.
    :return: Data where a random augmentation is applied per experiment.
    """

    augmented_data = np.empty(data.shape)
    for i, experiment in enumerate(data):
        augmented_data[i] = random.choice(augmentations).augment(experiment)
    return augmented_data
