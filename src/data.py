"""A module for preparing the training data for the baselines."""
from typing import List

import pandas as pd


def retrieve_instances_from_dataset(dataset: pd.DataFrame) -> List[str]:
    """Retrieve sentences with insertions from dataset.

    :param dataset: dataframe with labeled data
    :return: list of sentence strs
    """
    # fill the empty values with empty strings
    dataset = dataset.fillna("")

    instances = []

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )
            instances.append(sent_with_filler)

    return instances


def retrieve_labels_from_dataset_for_ranking(label_set: pd.DataFrame) -> List[float]:
    """Retrieve labels from dataset.

    :param label_set: dataframe with plausibility gold scores
    :return: list of rating floats
    """
    # the labels are already in the right order for the training instances, so we can just put them in a list
    return list(label_set["Label"])


def retrieve_labels_from_dataset_for_classification(
    label_set: pd.DataFrame,
) -> List[int]:
    """Retrieve labels from dataset.

    :param label_set: dataframe with class labels
    :return: list of int class labels 0, 1 or 2 (IMPLAUSIBLE, NEUTRAL, PLAUSIBLE)
    """
    # the labels are already in the right order for the training instances, so we can just put them in a list
    label_strs = list(label_set["Label"])
    label_ints = []

    for label_str in label_strs:
        if label_str == "IMPLAUSIBLE":
            label_ints.append(0)
        elif label_str == "NEUTRAL":
            label_ints.append(1)
        elif label_str == "PLAUSIBLE":
            label_ints.append(2)
        else:
            raise ValueError(f"Label {label_str} is not a valid plausibility class.")

    return label_ints
