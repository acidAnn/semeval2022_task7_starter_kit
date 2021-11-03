"""A module for checking the format of a submission with predictions.

The requirements for the dataframe representing a submission for the classification task are:
* shall have two columns, one with identifiers and another with class labels
* each id is a string:
    * it starts with an integer representing the id of the instance
    * next, there is an underscore
    * finally, the id of the filler (1 to 5)
    * e. g. "42_1" stands for the sentence with id 42 with filler 1
* the class label is string from the label set "IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"

The requirements for the dataframe representing a submission for the ranking task are:
* shall have two columns, one with identifiers and another with real-valued plausibility scores
* the id looks like the one for the classification task
* the plausibility score is a float
"""
import argparse
import logging
from typing import List

import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def check_format_of_submission(submission: pd.DataFrame, subtask: str) -> None:
    """Check the format of a dataframe with predictions.

    :param submission: dataframe with submission data
    :param subtask: str describing whether the submission is for the 'ranking' or 'classification' task
    """
    logging.debug(f"Verifying the format of a submission for subtask {subtask}")

    if subtask == "ranking":
        check_format_for_ranking_submission(submission)

    elif subtask == "classification":
        check_format_for_classification_submission(submission)

    else:
        raise ValueError(f"Evaluation mode {subtask} not available.")

    logging.debug("Format checking for submission successful. No problems detected.\n")


def check_format_for_ranking_submission(submission: pd.DataFrame) -> None:
    """Check the format of predictions for the ranking task.

    :param submission: dataframe with submission data
    """
    check_identifiers(submission["Id"])

    for rating_str in submission["Label"]:
        try:
            float(rating_str)
        except ValueError:
            raise ValueError(f"Rating {rating_str} is not a float.")


def check_format_for_classification_submission(submission: pd.DataFrame) -> None:
    """Check format of predictions for the classification task.

    :param submission: dataframe with submission data
    """
    check_identifiers(submission["Id"])

    valid_class_labels = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    for class_label in submission["Label"]:
        if class_label not in valid_class_labels:
            raise ValueError(
                f"Label {class_label} is not part of the label set {valid_class_labels}"
            )


def check_identifiers(id_list: List[str]) -> None:
    for identifier in id_list:
        if "_" not in identifier:
            raise ValueError(f"Id {identifier} does not contain an underscore.")
        else:
            sentence_id_str, filler_id_str = identifier.split("_")

            try:
                int(sentence_id_str)
            except ValueError:
                raise ValueError(
                    f"The sentence id {sentence_id_str} in id {identifier} is not a valid integer."
                )

            try:
                filler_id = int(filler_id_str)
            except ValueError:
                raise ValueError(
                    f"The filler id {filler_id_str} in id {identifier} is not a valid integer."
                )
            else:
                if 1 > filler_id or filler_id > 5:
                    raise ValueError(
                        f"The filler id {filler_id} in id {identifier} is not in the range of 1 to 5."
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check format of submission.")
    parser.add_argument(
        "--path_to_predictions",
        type=str,
        required=True,
        help="path to submission file with predictions",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        required=True,
        help="subtask: 'ranking' or 'classification'",
    )
    args = parser.parse_args()
    submission = pd.read_csv(
        args.path_to_predictions, delimiter="\t", header=None, names=["Id", "Label"]
    )
    check_format_of_submission(submission, args.subtask)
