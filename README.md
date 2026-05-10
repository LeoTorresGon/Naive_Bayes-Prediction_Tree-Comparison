# Social Media Impact Analysis: Predictive Modeling
A Machine Learning project focused on analyzing and predicting the overall impact of social media on users' lives. This project compares two classic classification algorithms: Gaussian Naive Bayes and Decision Trees.

## Overview
This project utilizes a dataset containing social media usage patterns and demographic information to classify the Overall_Impact of social media into three categories:

- Positive (1)
- Neutral (0)
- Negative (-1)

## Features & Preprocessing
The model processes the following features:

Demographics: Gender, Country, and Academic Level.
Usage: Most used platform and its perceived effect on academic performance.

### Preprocessing steps included:
Handling duplicates and missing values.
Label encoding for categorical variables (Gender, Academic Level, etc.).
Target mapping for the impact classes.

## Models Compared
1. Gaussian Naive Bayes
A probabilistic classifier based on Bayes' Theorem, assuming independence between features. It provides a baseline for how well the features predict the outcome based on probability distributions.

2. Decision Tree Classifier
A non-parametric supervised learning method. It creates a model that predicts the value of the target variable by learning simple decision rules inferred from the data features.


## Evaluation Metrics
The following metrics were implemented:

Accuracy Score: Overall percentage of correct predictions.
F1-Score (Weighted): Balance between precision and recall, accounting for class imbalance.
Cohen’s Kappa: Statistical measure of agreement between predicted and actual classifications.
Confusion Matrix: Detailed view of where the model is confusing specific classes.

## How to Run
Requirements:
- Python 3
```
pip install pandas numpy scikit-learn matplotlib
```
- Dataset: https://www.kaggle.com/datasets/abrerjawod/student-mental-health-and-burnout-dataset
  
Execution:
```
python preditor.py
```

## Results
The script outputs a comparative table and a visualization of the Decision Tree logic.
