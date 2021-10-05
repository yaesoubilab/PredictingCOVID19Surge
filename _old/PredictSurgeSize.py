from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

# use the entire dataset to predict the size of surge using linear regression and neural network

# read dataset
df = pd.read_csv('../outputs/prediction_datasets/data at week 82.0.csv')
# y_names
y_name_continues = 'Maximum hospitalization rate'
y_name_binary = 'If hospitalization threshold passed'
# feature names
column_names = df.columns.tolist()
column_names.remove(y_name_continues)
column_names.remove(y_name_binary)

# #####################
# # LINEAR REGRESSION #
# #####################
DEGREE_OF_POLYNOMIAL = 1
NUM_OF_FEATURES_WANTED = 10
FEATURE_SELECTION = 'lasso'  # method: 'rfe', or 'lasso', or 'pi'

data_lr = PreProcessor(df=df, feature_names=column_names, y_name=y_name_continues)

# pre-processing (standardization, add polynomial terms)
data_lr.preprocess(if_standardize=True, degree_of_polynomial=DEGREE_OF_POLYNOMIAL)

# feature selection
data_lr.feature_selection(estimator=linear_model.LinearRegression(),
                          method=FEATURE_SELECTION,
                          num_fs_wanted=NUM_OF_FEATURES_WANTED)
print(data_lr.featureName)
print('number of selected significant features', len(data_lr.featureName))

# train & predict
multi_linear_model = MultiLinearReg(df=data_lr.df, features=data_lr.featureName, y_name=data_lr.y_name)
multi_linear_model.run_many(num_bootstraps=10)

# performance
multi_linear_model.performancesTest.print()

##################
# Neural Network #
##################
NUM_OF_FEATURES_WANTED = 10
NUM_OF_NEURONS = 3 # default will be equal to the number of features + 2
ACTIVATION = 'logistic'
SOLVER = 'adam'
ALPHA = 0.0001  # L2 penalty (regularization term) parameter.
MAX_ITR = 1000

data_nn = PreProcessor(df=df, feature_names=column_names, y_name=y_name_continues)
# preprocess (standardization)
data_nn.preprocess(if_standardize=True)      # I did not add polynomial terms for neural network model
# feature selection
n_neurons = len(column_names) + 2 if NUM_OF_NEURONS is None else NUM_OF_NEURONS
data_nn.feature_selection(estimator=MLPRegressor(activation=ACTIVATION,
                                                 hidden_layer_sizes=(n_neurons,),
                                                 solver=SOLVER,
                                                 alpha=ALPHA,
                                                 max_iter=MAX_ITR
                                                 ),
                          method='pi',  # 'rfe' and 'lasso' is not applicable for NN
                          num_fs_wanted=NUM_OF_FEATURES_WANTED)
print(data_nn.featureName)
print('number of selected significant features', len(data_nn.featureName))

# train & predict
multi_nn_model = MultiNNRegression(df=data_nn.df, features=data_nn.featureName, y_name=data_nn.y_name)
multi_nn_model.run_many(num_bootstraps=100, activation=ACTIVATION, solver=SOLVER, alpha=ALPHA, max_iter=MAX_ITR)

# performance
multi_nn_model.performancesTest.print()


