from typing import NoReturn

from src.scripts.data_load import *
from src.scripts.data_preprocessing import *
from src.util.definitions import *


def apply_preprocess(
    data: np.ndarray, labels: List[str], data_scaler: StandardScaler
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Function to apply preprocessing on the data

    :param data: Data array
    :param labels: List of corresponding labels
    :param data_scaler: Scaler object which has stored training mean and variance
    :return: Preprocessed data and encoded labels
    """

    data_fixed_len = np.array(list(map(data_length_fix, data)))
    standardized_data = apply_standardize(data_fixed_len, data_scaler)
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

    train_data = wav2numpy(train_data_list)
    validation_data = wav2numpy(val_data_list)
    test_data = wav2numpy(test_data_list)
    data_scaler = fit_standardize(train_data)

    preprocessed_train_data, train_labels = apply_preprocess(
        train_data, train_labels_list, data_scaler
    )
    preprocessed_val_data, val_labels = apply_preprocess(
        validation_data, val_labels_list, data_scaler
    )
    preprocessed_test_data, test_labels = apply_preprocess(
        test_data, test_labels_list, data_scaler
    )

    balanced_train_data, train_data_labels = data_balancing(preprocessed_train_data, train_labels)

    if not save_path:
        save_path.mkdir(exist_ok=True, parents=True)

    np.save(str(PREPROCESSED_PATH / "train.npy"), balanced_train_data)
    np.save(str(PREPROCESSED_PATH / "val.npy"), preprocessed_val_data)
    np.save(str(PREPROCESSED_PATH / "test.npy"), preprocessed_test_data)
    np.save(str(PREPROCESSED_PATH / "train_labels.npy"), train_data_labels)
    np.save(str(PREPROCESSED_PATH / "val_labels.npy"), val_labels)
    np.save(str(PREPROCESSED_PATH / "test_labels.npy"), test_labels)
