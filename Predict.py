import pandas as pd

from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

feature_names = ['Obs: Incidence', 'Obs: Cumulative vaccination']
y_name_binary = 'if_surpass'
y_name_rate = 'maximum_occupancy'

# decision tree model
print('Decision trees')
Model1 = DecisionTree(features=feature_names, y_name=y_name_binary)
Model1.run(df=df, display_decision_path=True)

# linear regression model
print()
print('Linear regressions')
Model2 = LinearReg(features=feature_names, y_name=y_name_rate)
Model2.run(df=df)

