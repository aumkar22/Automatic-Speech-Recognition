import math
import random
import numpy as np

from typing import List, Optional, Tuple, NoReturn
from tensorflow.keras.utils import Sequence

from src.scripts.data_preprocessing import data_balancing
from src.scripts.augmenter import Augmentation, apply_augmentations
from src.scripts.data_preprocessing import mfcc_extractor


class BatchGenerator(Sequence):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        augmentations: Optional[List[Augmentation]] = None,
        balance: bool = True,
    ):
        """
        Batch generator based on Keras's Sequence class. This generates batches of size batch_size
        for both features as well as labels. Data can optionally be augmented and shuffled.

        :param features: Feature data with shape (experiments, samples, channels).
        :param labels: Label data with shape (experiments).
        :param batch_size: Size of batches.
        :param shuffle: Whether or not data should be shuffled.
        :param augmentations: Optional list of augmentations to be applied on a per-experiment
                              basis.
        :param balance: Balance training data batch.
        """
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations
        # Create indices for all experiments that will be used to select features/labels in
        # batches. Using a separate indices list also allows us to just shuffle this small list
        # when shuffling is enabled.
        self.indices = list(range(features.shape[0]))
        # Make sure we start shuffled if needed.
        self._shuffle_indices(self.shuffle)
        self.balance = balance

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, List[None]]:
        """
        Create, possibly augment, and return a batch of feature and label data.

        :param index: Index of the batch.
        :return: A tuple consisting of a batch of feature data and one of label data. The None List
                 is added to suppress a warning.
        """
        # Index can be interpreted as the batch number.
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indices = self.indices[start_index:end_index]

        feature_batch = self.features[batch_indices]
        label_batch = self.labels[batch_indices]

        if self.balance:
            feature_batch, label_batch = data_balancing(feature_batch, label_batch)
        if self.augmentations:
            feature_batch = apply_augmentations(feature_batch, self.augmentations)

        mfcc_feature_batch = np.array(list(map(mfcc_extractor, feature_batch)))
        mfcc_feature_batch = np.expand_dims(mfcc_feature_batch, axis=-1)

        # See https://stackoverflow.com/a/60131716 for the reason behind the [None].
        # IMPORTANT: [None] should be removed once TensorFlow is upgraded to v2.2.
        return mfcc_feature_batch, label_batch, [None]

    def __len__(self) -> int:
        """
        Calculate the number of batches that can be created using the provided batch size.

        :return: Number of batches.
        """
        return int(math.ceil(self.features.shape[0] / self.batch_size))

    def on_epoch_end(self) -> NoReturn:
        """
        Method that is called once training has finished an epoch. The only thing we need to do in
        those situations is shuffling the indices if that's been enabled in the constructor.

        :return: No return.
        """
        super().on_epoch_end()
        self._shuffle_indices(self.shuffle)

    def _shuffle_indices(self, shuffle) -> NoReturn:
        """
        Shuffle indices if shuffle is True. Shuffle has been added as an explicit parameter instead
        of just using self.shuffle to provide more intuition from callsites that shuffling is
        optional.

        :param shuffle: Whether or not data should be shuffled.
        :return: No return. Data is shuffled inplace.
        """
        if shuffle:
            random.shuffle(self.indices)
