import pandas as pd
from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *


# TODO: could we try fitting one NN using all features to predict the size of the peak?

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week 82.0.csv')
# y_names
y_name_continues = 'Maximum hospitalization rate'
y_name_binary = 'If hospitalization threshold passed'
# feature names
column_names = df.columns.tolist()
column_names.remove(y_name_continues)
column_names.remove(y_name_binary)

# standardize
data_nn = Dataframe(df=df, features=column_names, y_name=y_name_continues)
data_nn.preprocess(standardization=True)

# 1. fit the model using all features and all data
multi_nn_model = MultiNNRegression(df=data_nn.df,
                                   features=data_nn.features,
                                   y_name=data_nn.y_name)
multi_nn_model.run_many(num_bootstraps=10, max_iter=10000)

# 2. report the R2 using all data (no data is kept for testing)
multi_nn_model.performancesTest.print()

# 3. plot the predictive values vs. observed values
