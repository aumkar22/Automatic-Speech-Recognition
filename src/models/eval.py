import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from src.util.definitions import classes
from src.util.folder_check import *


class EvalVisualize(object):

    """
    Class to perform model evaluation and plot confusion matrix.
    """

    def __init__(self, ytrue: np.ndarray, ypred: np.ndarray):

        """
        Initialize with ground truth and prediction arrays.

        :param ytrue: Ground truth array
        :param ypred: Predictions
        """

        self.ytrue = ytrue
        self.ypred = ypred

    def get_metrics(self, save_path: Path, print_report: bool = False) -> NoReturn:

        """
        Classification report returns precision, recall and F1 scores for each class.

        :param save_path: Path to save metrics.
        :param print_report: Boolean parameter. Print report if True.
        :return: No return.
        """

        result = classification_report(
            self.ytrue, self.ypred, target_names=classes, digits=2, output_dict=True
        )
        report_df = pd.DataFrame.from_dict(result, orient="columns").dropna().round(2).T
        if print_report:
            print(report_df)

        path_check(save_path.parent, True)
        report_df.to_csv(str(save_path), index=False)

    def get_confusion_matrix(self, save_path: Path, plot_matrix: bool = False) -> NoReturn:

        """
        Function to plot confusion matrix.

        :param save_path: Path to save confusion matrix.
        :param plot_matrix: Boolean parameter. Plot if True.
        :return: No return.
        """

        cm = confusion_matrix(self.ytrue, self.ypred)
        normalized_cm = cm.astype("float") / cm.sum(axis=1)

        path_check(save_path.parent, True)

        plt.figure(figsize=(50, 50))
        sns.set(font_scale=1.4)
        sns.heatmap(
            np.round(normalized_cm, 2),
            annot=True,
            xticklabels=classes,
            yticklabels=classes,
            fmt="g",
        )
        plt.title("Normalized confusion matrix")
        plt.ylabel("True label", fontsize=30)
        plt.xlabel("Predicted label", fontsize=30)
        plt.savefig(save_path, dpi=250)

        if plot_matrix:
            plt.show()
