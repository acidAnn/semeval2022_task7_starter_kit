"""A module for scoring predictions according to the evaluation metrics."""
import argparse
import logging
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

from format_checker_for_submission import check_format_of_submission

logging.basicConfig(level=logging.DEBUG)


def score(submission_file: str, reference_file: str, subtask: str) -> float:
    """Assign an overall score to submitted predictions.

    :param submission_file: str path to submission file with predicted ratings
    :param reference_file: str path to file with gold ratings
    :param subtask: str indicating if the predictions are for the ranking or the classification task
    options: 'ranking' or 'classification'
    :return: float score
    """
    logging.debug(f"Subtask: {subtask}")
    logging.debug(f"Scoring submission in file {submission_file}")
    logging.debug(f"Compare to reference labels in file {reference_file}")
    predictions = []
    target = []

    submission = pd.read_csv(
        submission_file, sep="\t", header=None, names=["Id", "Label"]
    )
    check_format_of_submission(submission, subtask=subtask)

    reference = pd.read_csv(
        reference_file, sep="\t", header=None, names=["Id", "Label"]
    )
    # the reference file must have the same format as the submission file, so we use the same format checker
    check_format_of_submission(reference, subtask=subtask)

    if submission.size != reference.size:
        raise ValueError(
            "Submission does not contain the same number of rows as reference file."
        )

    for _, row in submission.iterrows():
        reference_indices = list(reference["Id"][reference["Id"] == row["Id"]].index)

        if not reference_indices:
            raise ValueError(
                f"Identifier {row['Id']} does not appear in reference file."
            )
        elif len(reference_indices) > 1:
            raise ValueError(
                f"Identifier {row['Id']} appears several times in reference file."
            )
        else:
            reference_index = reference_indices[0]

            if subtask == "ranking":
                target.append(float(reference["Label"][reference_index]))
                predictions.append(float(row["Label"]))
            elif subtask == "classification":
                target.append(reference["Label"][reference_index])
                predictions.append(row["Label"])
            else:
                raise ValueError(
                    f"Evaluation mode {subtask} not available: select ranking or classification"
                )

    if subtask == "ranking":
        score = spearmans_rank_correlation(
            gold_ratings=target, predicted_ratings=predictions
        )
        logging.debug(f"Spearman's rank correlation coefficient: {score}")

    elif subtask == "classification":
        prediction_ints = convert_class_names_to_int(predictions)
        target_ints = convert_class_names_to_int(target)

        score = accuracy_score(y_true=target_ints, y_pred=prediction_ints)
        logging.debug(f"Accuracy score: {score}")

    else:
        raise ValueError(
            f"Evaluation mode {subtask} not available: select ranking or classification"
        )

    return score


def convert_class_names_to_int(labels: List[str]) -> List[int]:
    """Convert class names to integer label indices.

    :param labels:
    :return:
    """
    class_names = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    label_indices = []

    for label in labels:
        try:
            label_index = class_names.index(label)
        except ValueError:
            raise ValueError(f"Label {label} is not in label set {class_names}.")
        else:
            label_indices.append(label_index)

    return label_indices


def spearmans_rank_correlation(
    gold_ratings: List[float], predicted_ratings: List[float]
) -> float:
    """Score submission for the ranking task with Spearman's rank correlation.

    :param gold_ratings: list of float gold ratings
    :param predicted_ratings: list of float predicted ratings
    :return: float Spearman's rank correlation coefficient
    """
    if len(gold_ratings) == 1 and len(predicted_ratings) == 1:
        raise ValueError("Cannot compute rank correlation on only one prediction.")

    return spearmanr(a=gold_ratings, b=predicted_ratings)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score submitted predictions.")
    parser.add_argument(
        "--path_to_predictions",
        type=str,
        required=True,
        help="path to submission file with predictions",
    )
    parser.add_argument(
        "--path_to_labels",
        type=str,
        required=True,
        help="path to reference file with gold labels",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        required=True,
        help="subtask: 'ranking' or 'classification'",
    )
    args = parser.parse_args()
    score(
        submission_file=args.path_to_predictions,
        reference_file=args.path_to_labels,
        subtask=args.subtask,
    )
