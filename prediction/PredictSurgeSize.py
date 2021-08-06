from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

# use the entire dataset to predict the size of surge using linear regression and neural network

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week 78.0.csv')
features = ['Obs: Incidence rate',
            'Obs: New hospitalization rate',
            'Obs: % of incidence due to Novel',
            'Obs: % of incidence due to Vaccinated',
            'Obs: % of new hospitalizations due to Novel',
            'Obs: % of new hospitalizations due to Vaccinated',
            'Obs: Prevalence susceptible',
            'Obs: Cumulative vaccination rate',
            'Obs: Cumulative hospitalization rate',
            'R0',
            'Duration of infectiousness-dominant',
            'Prob Hosp for 18-29',
            'Relative prob hosp by age-0',
            'Relative prob hosp by age-1',
            'Relative prob hosp by age-2',
            'Relative prob hosp by age-4',
            'Relative prob hosp by age-5',
            'Relative prob hosp by age-6',
            'Relative prob hosp by age-7',
            'Ratio of hospitalization probability by profile-1',
            'Ratio of hospitalization probability by profile-2',
            'Ratio transmissibility by profile-1',
            'Ratio transmissibility by profile-2',
            'Ratio of infectiousness duration by profile-1',
            'Ratio of infectiousness duration by profile-2',
            'Duration of R-0',
            'Duration of R-1',
            'Duration of R-2',
            'Duration of vaccine immunity',
            'Prob novel strain params-0',
            'Prob novel strain params-1',
            'Prob novel strain params-2',
            'Vaccine effectiveness against infection with novel',
            'PD Y1 thresholds-0',
            'PD Y1 thresholds-1',
            'Change in contacts - PD Y1',
            'Change in contacts - PD Y1+'
            ]
y_name = 'Maximum hospitalization rate'

# #####################
# # LINEAR REGRESSION #
# #####################
DEGREE_OF_POLYNOMIAL = 1
NUM_OF_FEATURES_WANTED = 10
FEATURE_SELECTION = 'lasso'  # method: 'rfe', or 'lasso', or 'pi'

df_lr = Dataframe(df=df, features=features, y_name=y_name)

# pre-processing (standardization, add polynomial terms)
df_lr.preprocess(standardization=True, degree_of_polynomial=DEGREE_OF_POLYNOMIAL)

# feature selection
df_lr.feature_selection(estimator=linear_model.LinearRegression(),
                        method=FEATURE_SELECTION,
                        num_fs_wanted=NUM_OF_FEATURES_WANTED)
print(df_lr.selected_features)
print('number of selected significant features', len(df_lr.selected_features))

# train & predict
multi_linear_model = MultiLinearReg(df=df_lr.df, features=df_lr.selected_features, y_name=df_lr.y_name)
multi_linear_model.run_many(num_bootstraps=100)

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

df_nn = Dataframe(df=df, features=features, y_name=y_name)
# preprocess (standardization)
df_nn.preprocess(standardization=True)      # I did not add polynomial terms for neural network model
# feature selection
n_neurons = len(features) + 2 if NUM_OF_NEURONS is None else NUM_OF_NEURONS
df_nn.feature_selection(estimator=MLPRegressor(activation=ACTIVATION,
                                               hidden_layer_sizes=(n_neurons,),
                                               solver=SOLVER,
                                               alpha=ALPHA,
                                               max_iter=MAX_ITR
                                               ),
                        method='pi',    # 'rfe' and 'lasso' is not applicable for NN
                        num_fs_wanted=NUM_OF_FEATURES_WANTED)
print(df_nn.selected_features)
print('number of selected significant features', len(df_nn.selected_features))

# train & predict
multi_nn_model = MultiNNRegression(df=df_nn.df, features=df_nn.selected_features, y_name=df_nn.y_name)
multi_nn_model.run_many(num_bootstraps=100, activation=ACTIVATION, solver=SOLVER, alpha=ALPHA, max_iter=MAX_ITR)

# performance
multi_nn_model.performancesTest.print()


