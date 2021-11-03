"""A module for preparing the training data for the baselines."""
import logging
from typing import List, Tuple

import pandas as pd


def retrieve_instances_from_dataset(
    dataset: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Retrieve sentences with insertions from dataset.

    :param dataset: dataframe with labeled data
    :return: a tuple with
    * a list of id strs
    * a list of sentence strs
    """
    # fill the empty values with empty strings
    dataset = dataset.fillna("")

    ids = []
    instances = []

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            ids.append(f"{row['Id']}_{filler_index}")

            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )
            instances.append(sent_with_filler)

    return ids, instances


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


def write_predictions_to_file(
    path_to_predictions: str, ids: List[str], predictions: List, subtask: str
) -> pd.DataFrame:
    """Write the instance indices and predictions to a tsv file.

    :param path_to_predictions: str path to file where to write the predictions
    :param ids: list of str instance indices
    :param predictions: list of predictions
    :param subtask: str indicating "ranking" or "classification"
    :return: pandas dataframe with ids and predictions
    """
    if subtask == "classification":
        predictions = convert_class_indices_to_labels(predictions)

    dataframe = pd.DataFrame({"Id": ids, "Label": predictions})
    logging.info(f"--> Writing predictions to {path_to_predictions}")
    dataframe.to_csv(path_to_predictions, sep="\t", index=False, header=False)

    return dataframe


def convert_class_indices_to_labels(class_indices: List[int]) -> List[str]:
    """Convert integer class indices to str labels.

    :param class_indices: list of int class indices (0 to 2)
    :return: list of label strs from set "IMPLAUSIBLE" / "NEUTRAL" / "PLAUSIBLE"
    """
    labels = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    return [labels[class_index] for class_index in class_indices]
