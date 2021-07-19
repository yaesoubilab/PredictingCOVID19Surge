import pandas as pd

from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

POLYNOMIAL_DEGREE = 2
features = ['Obs: New hospitalizations',
            'Obs: Cumulative vaccination rate', 'Obs: Cumulative hospitalizations',
            'R0s-0']

y_binary = 'If hospitalization threshold passed'
y_continuous = 'Maximum hospitalization rate'

# TODO: since we will be trying a large number of features,
#   can we implement l1 or l2 regularization for the logistic regression and the linear regression?
#   ( I think you can run Ridge and Lasso models for the linear regression models:
#   https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

Model = LinearReg(df=df, features=features, y_name=y_continuous)
Model.run(degree_of_polynomial=POLYNOMIAL_DEGREE)

# TODO: would you also add the logistic regression and the decision tree models?

# TODO: in terms of performance measures, would you please report the mean squared error for
#  linear regression models?

# TODO: Finally, would you please add the bootstrap algorithm (not the optimism-corrected one,
#   but the one we had initially implemented: keep 20% of data for testing and repeating
#   the model training-testing 200 times to form confidence intervals)?
