import pandas as pd
from sklearn.neural_network import MLPRegressor

from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

import SimPy.Statistics as Stat

NUM_OF_NEURONS = None
DECIMAL = 4     # decimal place for cross-validation scores
CV = 30          # num of splits for cross validation
week = '86'
list_of_num_features_wanted = [10, 20, 30, 40]
list_of_alphas = [0.0001, 0.001, 0.01, 1]

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week {}.0.csv'.format(week))
# randomize rows
df = df.sample(frac=1, random_state=10)
# y_names
y_name_continues = 'Maximum hospitalization rate'
y_name_binary = 'If hospitalization threshold passed'
# feature names
feature_names = df.columns.tolist()
feature_names.remove(y_name_continues)
feature_names.remove(y_name_binary)

# preprocess (standardize)
data = Dataframe(df=df, features=feature_names, y_name=y_name_continues)
data.preprocess(if_standardize=True)

# CV to choose alpha + number of features
n_neurons = len(feature_names) + 2 if NUM_OF_NEURONS is None else NUM_OF_NEURONS
list_cv_score = [list_of_num_features_wanted]
for ALPHA in list_of_alphas:
    cv_score_per_alpha = []
    for NUM_FEATURES in list_of_num_features_wanted:
        print('alpha:', ALPHA, '; num_features:', NUM_FEATURES)
        # model hyper-parameters
        model = MLPRegressor(alpha=ALPHA, hidden_layer_sizes=(n_neurons, ),
                             max_iter=1000, solver='sgd', activation='logistic')
        # feature selection
        data.feature_selection(estimator=model, method='pi', num_fs_wanted=NUM_FEATURES)
        # cross-validation
        stat_cv = Stat.SummaryStat(name='cross-validation score for one alpha-feature combination',
                                   data=cross_val_score(estimator=model, X=data.X, y=data.y.ravel(), cv=CV))
        cv_score_per_alpha.append(stat_cv.get_formatted_mean_and_interval(deci=DECIMAL, interval_type="p"))
    list_cv_score.append(cv_score_per_alpha)

string_of_alphas = ['alpha_{}'.format(alpha) for alpha in list_of_alphas]
string_of_alphas.insert(0, 'num_features')
cv_df = pd.DataFrame(np.asarray(list_cv_score).T, columns=string_of_alphas)

print('\n', cv_df)
cv_df.to_csv('../outputs/cv_score/cv_score_{}fold_week{}.csv'.format(CV, week))


# # # feature selection
# # data_nn.feature_selection(estimator=MLPRegressor(activation=ACTIVATION,
# #                                                  hidden_layer_sizes=(n_neurons,),
# #                                                  solver=SOLVER,
# #                                                  alpha=ALPHA,
# #                                                  max_iter=MAX_ITR),
# #                           method='pi',
# #                           num_fs_wanted=NUM_OF_FEATURES_WANTED)
#
# # 1. fit the model using all features and all data
# multi_nn_model = MultiNNRegression(df=data_nn.df,
#                                    features=data_nn.features,
#                                    y_name=data_nn.y_name)
# multi_nn_model.run_many(num_bootstraps=10,
#                         max_iter=10000,
#                         activation=ACTIVATION,
#                         solver=SOLVER,
#                         test_size=0.2)
#
# # 2. report the R2 using all data (no data is kept for testing)
# multi_nn_model.performancesTest.print()
