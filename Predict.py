import pandas as pd
from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

POLYNOMIAL_DEGREE = 2
features = ['Obs: New hospitalizations',
            'Obs: Cumulative vaccination rate', 'Obs: Cumulative hospitalizations',
            'R0s-0']

y_binary = 'If hospitalization threshold passed'
y_continuous = 'Maximum hospitalization rate'

Model = LinearReg(features=features, y_name=y_continuous)
Model.run(df=df, degree_of_polynomial=POLYNOMIAL_DEGREE)
