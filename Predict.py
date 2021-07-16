import pandas as pd

from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

# TODO: since we will be trying a large number of features, we also need to think about feature selection
#   For now, maybe we just implemented l1 or l2 regularization for the logistic regression and the linear regression.
#   (you can run Ridge and Lasso models: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)


# TODO: feature_names should be updated (see my comments on BuildDataset.py)
feature_names = ['Obs: Incidence', 'Obs: Cumulative vaccination']
y_name_binary = 'If hospitalization threshold passed'
y_name_rate = 'Maximum hospitalization rate'

# TODO: would you please also add a logistic regression model?
# decision tree model
print('Decision trees')
Model1 = DecisionTree(features=feature_names, y_name=y_name_binary)
Model1.run(df=df, display_decision_path=False)

# linear regression model
print()
print('Linear regressions')
Model2 = LinearReg(features=feature_names, y_name=y_name_rate)
Model2.run(df=df)

# TODO: in terms of performance measures, would you please report the following?
#   For classifiers: ACU_ROC on the test set (it doesn't have to be optimism-corrected)
#   For linear regression models: Mean squared error and R2 on the test set
