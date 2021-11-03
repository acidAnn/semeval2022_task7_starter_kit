"""A module for running baseline models.

Examples:
python main.py --path_to_train train_data.tsv --path_to_training_labels train_labels.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_labels.tsv --path_to_predictions pred_dev_class.tsv --classification_baseline bag-of-words
python main.py --path_to_train train_data.tsv --path_to_training_labels train_scores.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_scores.tsv --path_to_predictions pred_dev_rank.tsv --ranking_baseline bag-of-words
"""
import argparse
import logging

import pandas as pd

from data import (
    retrieve_instances_from_dataset,
    retrieve_labels_from_dataset_for_classification,
    retrieve_labels_from_dataset_for_ranking,
    write_predictions_to_file,
)
from format_checker_for_dataset import check_format_of_dataset
from format_checker_for_submission import check_format_of_submission
from models import BowClassificationBaseline, BowRankingBaseline
from scorer import score

logging.basicConfig(level=logging.DEBUG)

# Initialize argparse
ap = argparse.ArgumentParser(description="Run baselines for SemEval-2022 Task 7.")
ap.add_argument(
    "--path_to_train",
    required=True,
    type=str,
    help="Path to the training instances in tsv format.",
)
ap.add_argument(
    "--path_to_training_labels",
    required=True,
    type=str,
    help="Path to the labels for the training instances in tsv format.",
)
ap.add_argument(
    "--path_to_dev",
    required=True,
    type=str,
    help="Path to the dev instances in tsv format.",
)
ap.add_argument(
    "--path_to_dev_labels",
    required=True,
    type=str,
    help="Path to the labels for the dev instances in tsv format.",
)
ap.add_argument(
    "--path_to_predictions",
    required=True,
    type=str,
    help="Path to tsv file in which to write the predictions",
)
ap.add_argument(
    "--classification_baseline",
    type=str,
    help="Select a baseline classifier: bag-of-words",
)
ap.add_argument(
    "--ranking_baseline",
    type=str,
    help="Select a baseline ranking model: bag-of-words",
)
args = vars(ap.parse_args())

logging.debug(f"Read training dataset from file {args['path_to_train']}")
train_set = pd.read_csv(args["path_to_train"], sep="\t", quoting=3)
check_format_of_dataset(train_set)
_, training_instances = retrieve_instances_from_dataset(train_set)

logging.debug(f"Read dev dataset from file {args['path_to_dev']}")
dev_set = pd.read_csv(args["path_to_dev"], sep="\t", quoting=3)
check_format_of_dataset(dev_set)
dev_ids, dev_instances = retrieve_instances_from_dataset(dev_set)

# Run the baseline
if args["classification_baseline"] or args["ranking_baseline"]:
    subtask = "classification" if args["classification_baseline"] else "ranking"

    logging.debug(
        f"Read gold labels for training dataset from file {args['path_to_training_labels']}"
    )
    training_label_set = pd.read_csv(
        args["path_to_training_labels"], sep="\t", header=None, names=["Id", "Label"]
    )
    check_format_of_submission(training_label_set, subtask=subtask)
    baseline_model = None

    if (
        args["classification_baseline"]
        and args["classification_baseline"] == "bag-of-words"
    ):
        logging.debug("Subtask A: multi-class classification")
        logging.debug("Run classification baseline with bag of words")
        baseline_model = BowClassificationBaseline()
        training_labels = retrieve_labels_from_dataset_for_classification(
            training_label_set
        )

    elif args["ranking_baseline"] and args["ranking_baseline"] == "bag-of-words":
        logging.debug("Subtask B: ranking")
        logging.debug("Run ranking baseline with bag of words")
        baseline_model = BowRankingBaseline()
        training_labels = retrieve_labels_from_dataset_for_ranking(training_label_set)

    dev_predictions = baseline_model.run_held_out_evaluation(
        training_instances=training_instances,
        training_labels=training_labels,
        dev_instances=dev_instances,
    )
    prediction_dataframe = write_predictions_to_file(
        path_to_predictions=args["path_to_predictions"],
        ids=dev_ids,
        predictions=dev_predictions,
        subtask=subtask,
    )

    logging.debug("Score predictions for dev set")
    score(
        submission_file=args["path_to_predictions"],
        reference_file=args["path_to_dev_labels"],
        subtask=subtask,
    )
