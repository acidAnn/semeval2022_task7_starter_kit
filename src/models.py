"""A module with the baseline models for the classification and ranking subtasks."""
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline

from scorer import spearmans_rank_correlation


def identity(x):
    return x


class BowClassificationBaseline:
    """A baseline for the classification task that combines tf-idf feature extraction and multinomial Naive Bayes."""

    def __init__(self):
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        self.model = Pipeline([("vec", vec), ("cls", MultinomialNB())])

    def run_cross_validation(
        self, instances: List[str], labels: List[int]
    ) -> List[float]:
        """Run k-fold cross-validation on the input data.

        :param instances: list of str instances, i. e. sentences with insertions
        :param labels: list of float gold ratings
        :return: a list of with the float accuracy for each run
        """
        instances = [text.split() for text in instances]
        cv = cross_val_score(self.model, instances, labels, cv=5, scoring="accuracy")
        return list(cv)

    def run_held_out_evaluation(
        self,
        training_instances: List[str],
        training_labels: List[int],
        dev_instances: List[str],
        dev_labels: List[int],
    ):
        self.model.fit(X=training_instances, y=training_labels)
        predictions = self.model.predict(dev_instances)
        return accuracy_score(gold_ratings=dev_labels, predicted_ratings=predictions)


class BowRankingBaseline:
    """A baseline for the ranking task that combines tf-idf feature extraction and linear regression."""

    def __init__(self):
        self.model = Pipeline(
            [("vec", TfidfVectorizer()), ("regr", LinearRegression())]
        )

    def run_cross_validation(
        self, instances: List[str], labels: List[float]
    ) -> List[float]:
        """Run k-fold cross-validation on the input data.

        :param instances: list of str instances, i. e. sentences with insertions
        :param labels: list of float gold ratings
        :return: a list of with the float Spearman's rank correlation coefficient for each run
        """
        scorer = make_scorer(spearmans_rank_correlation, greater_is_better=True)
        scores_per_run = cross_val_score(
            self.model, instances, labels, cv=5, scoring=scorer
        )
        return list(scores_per_run)

    def run_held_out_evaluation(
        self,
        training_instances: List[str],
        training_labels: List[float],
        dev_instances: List[str],
        dev_labels: List[float],
    ):
        self.model.fit(X=training_instances, y=training_labels)
        predictions = self.model.predict(dev_instances)
        return spearmans_rank_correlation(
            gold_ratings=dev_labels, predicted_ratings=predictions
        )
