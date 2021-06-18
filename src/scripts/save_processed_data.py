from typing import NoReturn

from src.scripts.data_load import *
from src.scripts.data_preprocessing import *
from src.util.definitions import *


def apply_preprocess(
    data: np.ndarray, labels: List[str], data_scaler: StandardScaler, train: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to apply preprocessing on the data

    :param data: Data array
    :param labels: List of corresponding labels
    :param data_scaler: Scaler object which has stored training mean and variance
    :param train: Boolean parameter. True if data = training data. Apply length fix only for
    validation and test data. Train data length is already fixed for obtaining StandardScaler
    object.
    :return: Preprocessed data and encoded labels
    """

    if not train:
        data = np.array(list(map(data_length_fix, data)))

    standardized_data = apply_standardize(data, data_scaler)
    encoded_label = data_encode(labels)

    return standardized_data, encoded_label


def save_data(save_path: Path) -> NoReturn:

    """
    Function to save preprocessed data as npy files

    :param save_path: Path where data should be stored
    :return: No return
    """

    val_data_list, val_labels_list = get_test_val_labels_list(val_list_from_df)
    test_data_list, test_labels_list = get_test_val_labels_list(test_list_from_df)
    train_data_list, train_labels_list = train_data_labels_list(test_data_list, val_data_list)

    train_data, train_indices_to_ignore = wav2numpy(train_data_list)
    validation_data, val_indices_to_ignore = wav2numpy(val_data_list)
    test_data, test_indices_to_ignore = wav2numpy(test_data_list)

    train_fixed_len = np.array(list(map(data_length_fix, train_data)))
    data_scaler = fit_standardize(train_fixed_len)

    train_labels_list_after_excluded_indices = [
        label
        for index, label in enumerate(train_labels_list)
        if index not in train_indices_to_ignore
    ]
    val_labels_list_after_excluded_indices = [
        label for index, label in enumerate(val_labels_list) if index not in val_indices_to_ignore
    ]
    test_labels_list_after_excluded_indices = [
        label
        for index, label in enumerate(test_labels_list)
        if index not in test_indices_to_ignore
    ]

    preprocessed_train_data, train_labels = apply_preprocess(
        train_fixed_len, train_labels_list_after_excluded_indices, data_scaler, True
    )
    preprocessed_val_data, val_labels = apply_preprocess(
        validation_data, val_labels_list_after_excluded_indices, data_scaler, False
    )
    preprocessed_test_data, test_labels = apply_preprocess(
        test_data, test_labels_list_after_excluded_indices, data_scaler, False
    )

    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)

    np.save(str(save_path / "train.npy"), preprocessed_train_data)
    np.save(str(save_path / "val.npy"), preprocessed_val_data)
    np.save(str(save_path / "test.npy"), preprocessed_test_data)
    np.save(str(save_path / "train_labels.npy"), train_labels)
    np.save(str(save_path / "val_labels.npy"), val_labels)
    np.save(str(save_path / "test_labels.npy"), test_labels)
