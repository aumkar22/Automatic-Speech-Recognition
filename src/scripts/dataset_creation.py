import tensorflow as tf
import numpy as np

from typing import Optional, List

from src.scripts.augmenter import Augmentation
from src.scripts.generator import BatchGenerator


def create_dataset_generator(
    datax: np.ndarray,
    datay: np.ndarray,
    train: bool,
    augmentations: Optional[List[Augmentation]],
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Function to create tensorflow dataset from batch generator

    :param datax: Data features
    :param datay: Data labels
    :param train: True for training, False otherwise
    :param augmentations: List of augmentations for training, None otherwise
    :param batch_size: Batch size
    :return: Tensorflow dataset
    """

    data_gen = BatchGenerator(datax, datay, augmentations=augmentations)
    mfcc_dataset = tf.data.Dataset.from_generator(
        lambda: map(tuple, data_gen), output_types=(tf.float64, tf.int64)
    )
    mfcc_dataset.cache()
    if train:
        mfcc_dataset.shuffle(buffer_size=len(datax))
    mfcc_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    mfcc_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return mfcc_dataset
