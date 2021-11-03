# :bulb: SemEval-2022 Task 7: Identifying Plausible Clarifications of Implicit and Underspecified Phrases in Instructional Texts
Starter kit with a format checker, a scorer and baselines

## Contents
- [The Task](https://github.com/acidAnn/semeval2022_task7_starter_kit#the-task)
- [Repo Setup](https://github.com/acidAnn/semeval2022_task7_starter_kit#repo-setup)
- [Format Checker](https://github.com/acidAnn/semeval2022_task7_starter_kit#format-checker)
- [Scorer](https://github.com/acidAnn/semeval2022_task7_starter_kit#scorer)
- [Baselines](https://github.com/acidAnn/semeval2022_task7_starter_kit#baselines)
- [Questions? Need a Clarification?](https://github.com/acidAnn/semeval2022_task7_starter_kit/tree/dev_set#questions-need-a-clarification-)

## The Task
The goal of this shared task is to evaluate the ability of NLP systems to distinguish between **plausible and implausible clarifications** of an instruction. 

For details on the task, see [the task website](https://clarificationtask.github.io) and [the CodaLab page](https://competitions.codalab.org/competitions/35210).

## Repo Setup
After cloning this repository, install the dependencies by running the following command:
```shell
$ pip install -r requirements.txt
```

To download the training data and the corresponding labels, got to [the task website](https://clarificationtask.github.io):
* one file with the training instances
* one reference file with the gold class labels for these instances (classification subtask)
* one reference file with the gold plausibility scores for these instances (ranking subtask)

## Format Checker
Each submission file with predictions has to be a TSV.

The requirements for a classification task submission are:
* there should be two columns, one for identifiers and another for the class labels
* the id is a string:
    * it starts with an integer representing the id of the instance
    * next, there is an underscore
    * finally, the id of the filler (1 to 5)
    * e. g. "42_1" stands for the sentence with id 42 with filler 1
* the class label is string from the label set "IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"

The requirements for a ranking task submission are:
* there should be two columns, one for identifiers and another for the plausibility scores
* the id looks like the one for the classification task
* the plausibility score is a float

To see an example for these format requirements, have a look at the files with gold class labels and plausibility scores published on [the task website](https://clarificationtask.github.io).

There is a format checker for submission files. You can call it with the following flags:
```shell
# for the classification task
$ python format_checker_for_submission.py --path_to_predictions submission_classification.tsv --subtask classification
# for the ranking task
$ python format_checker_for_submission.py --path_to_predictions submission_ranking.tsv --subtask ranking
```

## Scorer
The scorer takes 
* a submission file with predictions in the above mentioned format 
* a reference file with gold labels that is in the same format as the submission file 

You can download the reference files for the two subtasks from [the task website](https://clarificationtask.github.io).

For the classification task, the scorer calculates the accuracy:
```shell
$ python scorer.py --path_to_predictions submission_classification.tsv --path_to_labels reference_classification.tsv --subtask classification
```

For the ranking task, the scorer computes Spearman's rank correlation coefficient:
```shell
$ python scorer.py --path_to_predictions submission_ranking.tsv --path_to_labels reference_ranking.tsv --subtask ranking
```

## Baselines
We provide two very simple baseline models here to help you to get started with the shared task.
The baseline for the classification subtask combines a tf-idf feature extractor with a Naive Bayes classifier.
The baseline for the ranking subtask uses a tf-idf feature extractor and a linear regression model.

After being trained on the training set, these models achieve the following performance on the development set:

|subtask and performance metric|performance|
|---|---|
|multi-class classification: accuracy |37.68%|
|ranking:  Spearman's rank correlation coefficient|+0.0560|

You can use the script `main.py` to reproduce these training and evaluation steps:
```shell
# for the classification subtask
$ python main.py --path_to_train train_data.tsv --path_to_training_labels train_labels.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_labels.tsv --path_to_predictions pred_dev_class.tsv --classification_baseline bag-of-words
# for the ranking subtask
$ python main.py --path_to_train train_data.tsv --path_to_training_labels train_scores.tsv --path_to_dev dev_data.tsv --path_to_dev_labels dev_scores.tsv --path_to_predictions pred_dev_rank.tsv --ranking_baseline bag-of-words
```
The script produces a TSV file with predictions (under the path specified with the flag `path_to_predictions`) and then hands that file to the scorer script.

## Questions? Need a Clarification? :)
If you have technical trouble with the code in this repo, please open an [Issue](https://github.com/acidAnn/semeval2022_task7_starter_kit/issues).
You can also ask question on the task in general in our [Google group](https://groups.google.com/g/semeval2022-task7/).