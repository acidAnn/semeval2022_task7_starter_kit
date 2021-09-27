"""A module with the baseline models for the classification and ranking subtasks."""
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
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
