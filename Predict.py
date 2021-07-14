import pandas as pd

from covid_prediction.DataDirectory import *
from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

# TODO: see my comments on feature_engineering.py first.
#   Since we are also interested in predicting the maximum rate of hospitalization, would you
#   also add a linear regression model too?

Model = DecisionTree(features=FEATURES, y_name=Y_NAME)
Model.run(df=df)
