import pandas as pd
from covid_prediction.prediction_models import *
from covid_prediction.DataDirectory import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

Model = DecisionTree(features=FEATURES, y_name=Y_NAME)
Model.run(df=df)
