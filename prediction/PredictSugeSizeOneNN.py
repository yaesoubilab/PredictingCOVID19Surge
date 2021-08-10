import pandas as pd
from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

ACTIVATION = 'logistic'
SOLVER = 'sgd'
NUM_OF_NEURONS = None
ALPHA = 0.0001  # L2 penalty (regularization term) parameter.
MAX_ITR = 1000
NUM_OF_FEATURES_WANTED = 10

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
data_nn.preprocess(if_standardize=True)

# # feature selection
# n_neurons = len(column_names) + 2 if NUM_OF_NEURONS is None else NUM_OF_NEURONS
# data_nn.feature_selection(estimator=MLPRegressor(activation=ACTIVATION,
#                                                  hidden_layer_sizes=(n_neurons,),
#                                                  solver=SOLVER,
#                                                  alpha=ALPHA,
#                                                  max_iter=MAX_ITR),
#                           method='pi',
#                           num_fs_wanted=NUM_OF_FEATURES_WANTED)

# 1. fit the model using all features and all data
multi_nn_model = MultiNNRegression(df=data_nn.df,
                                   features=data_nn.features,
                                   y_name=data_nn.y_name)
multi_nn_model.run_many(num_bootstraps=10,
                        max_iter=10000,
                        activation=ACTIVATION,
                        solver=SOLVER,
                        test_size=0.2)

# 2. report the R2 using all data (no data is kept for testing)
multi_nn_model.performancesTest.print()
