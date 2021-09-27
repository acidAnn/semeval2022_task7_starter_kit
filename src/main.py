"""A module for running baseline models.

Examples:
python main.py --path_to_train train_data.tsv --path_to_labels train_labels.tsv --classification_baseline bag-of-words
python main.py --path_to_train train_data.tsv --path_to_labels train_scores.tsv --ranking_baseline bag-of-words
"""
import argparse
import logging
import statistics

import pandas as pd

from data import (
    retrieve_instances_from_dataset,
    retrieve_labels_from_dataset_for_classification,
    retrieve_labels_from_dataset_for_ranking,
)
from format_checker_for_training_dataset import check_format_of_training_dataset
from format_checker_for_submission import check_format_of_submission
from models import BowClassificationBaseline, BowRankingBaseline

logging.basicConfig(level=logging.DEBUG)

# Initialize argparse.
ap = argparse.ArgumentParser(description="Run baselines for SemEval-2022 Task 7.")
ap.add_argument(
    "--path_to_train",
    required=True,
    type=str,
    help="Path to the training instances in tsv format.",
)
ap.add_argument(
    "--path_to_labels",
    required=True,
    type=str,
    help="Path to the labels for the training instances in tsv format.",
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

# Read the data
logging.debug(f"Read training dataset from file {args['path_to_train']}")
train_set = pd.read_csv(args["path_to_train"], sep="\t", quoting=3)
check_format_of_training_dataset(train_set)

# Run the baseline
if args["classification_baseline"]:
    logging.debug("Subtask A: multi-class classification")
    logging.debug(
        f"Read class labels for training dataset from file {args['path_to_labels']}"
    )
    label_set = pd.read_csv(
        args["path_to_labels"], sep="\t", header=None, names=["Id", "Label"]
    )
    check_format_of_submission(label_set, subtask="classification")

    instances = retrieve_instances_from_dataset(train_set)
    labels = retrieve_labels_from_dataset_for_classification(label_set)

    # Select a classifier
    if args["classification_baseline"] == "bag-of-words":
        logging.debug(
            "Run classification baseline with bag of words: 5-fold cross-validation"
        )
        classification_baseline = BowClassificationBaseline()
        scores_per_run = classification_baseline.run_cross_validation(
            instances=instances, labels=labels
        )
        logging.debug(f"Accuracy per cross-validation run: {scores_per_run}")
        logging.debug(f"Mean across all runs: {statistics.mean(scores_per_run)}")
        logging.debug(f"Standard deviation: {statistics.stdev(scores_per_run)}")

elif args["ranking_baseline"]:
    logging.debug("Subtask B: ranking")
    logging.debug(
        f"Read plausibility scores for training dataset from file {args['path_to_labels']}"
    )
    label_set = pd.read_csv(
        args["path_to_labels"], sep="\t", header=None, names=["Id", "Label"]
    )
    check_format_of_submission(label_set, subtask="ranking")

    instances = retrieve_instances_from_dataset(train_set)
    labels = retrieve_labels_from_dataset_for_ranking(label_set)

    if args["ranking_baseline"] == "bag-of-words":
        logging.debug(
            "Run ranking baseline with linear regression: 5-fold cross-validation"
        )
        ranking_baseline = BowRankingBaseline()
        scores_per_run = ranking_baseline.run_cross_validation(instances, labels)
        logging.debug(
            f"Spearman's rank correlation per cross-validation run: {scores_per_run}"
        )
        logging.debug(f"Mean across all runs: {statistics.mean(scores_per_run)}")
        logging.debug(f"Standard deviation: {statistics.stdev(scores_per_run)}")
