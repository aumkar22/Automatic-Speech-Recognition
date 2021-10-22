import math
import numpy as np
import tensorflow as tf

from typing import List, Optional, Tuple
from tensorflow.keras.utils import Sequence

from src.scripts.augmenter import Augmentation, apply_augmentations
from src.scripts.data_preprocessing import mfcc_extractor


class BatchGenerator(Sequence):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        augmentations: Optional[List[Augmentation]] = None,
    ):
        """
        Batch generator based on Keras's Sequence class. This generates batches of size batch_size
        for both features as well as labels. Data can optionally be augmented and shuffled.

        :param features: Feature data with shape (experiments, samples, channels).
        :param labels: Label data with shape (experiments).
        :param batch_size: Size of batches.
        :param augmentations: Optional list of augmentations to be applied on a per-experiment
                              basis.
        """
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        # Create indices for all experiments that will be used to select features/labels in
        # batches.
        self.indices = list(range(features.shape[0]))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
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
        label_batch_tensor = tf.convert_to_tensor(label_batch, dtype=tf.int64)

        # if self.balance:
        #     feature_batch, label_batch = data_balancing(feature_batch, label_batch)
        if self.augmentations:
            feature_batch = apply_augmentations(feature_batch, self.augmentations)

        mfcc_feature_batch = np.array(list(map(mfcc_extractor, feature_batch)))
        mfcc_feature_batch_tensor = tf.convert_to_tensor(mfcc_feature_batch, dtype=tf.float64)
        mfcc_feature_batch_tensor = tf.expand_dims(mfcc_feature_batch_tensor, axis=-1)
        return mfcc_feature_batch_tensor, label_batch_tensor

    def __len__(self) -> int:
        """
        Calculate the number of batches that can be created using the provided batch size.

        :return: Number of batches.
        """
        return int(math.ceil(self.features.shape[0] / self.batch_size))
