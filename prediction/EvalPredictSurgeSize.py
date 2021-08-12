import pandas as pd

import covid_prediction.evaluate_predictive_models as E

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week 82.0.csv')
# y_names
y_name_continues = 'Maximum hospitalization rate'
y_name_binary = 'If hospitalization threshold passed'
# feature names
feature_names = df.columns.tolist()
feature_names.remove(y_name_continues)
feature_names.remove(y_name_binary)

OUTCOME = 'Maximum hospitalization rate'
# NUM_OF_BOOTSTRAPS = 10
POLY_DEGREE = 1
IF_STANDARDIZED = True
PENALTY = 'l2'  # 'l1', 'l2', or 'none'

# settings for NN
decimal = 4     # decimal place for cross-validation scores
cv = 10         # num of splits for cross validation
feature_selection = 'pi'
list_of_num_features_wanted = [10, 20, 30, 40]
list_of_alphas = [0.0001, 0.001, 0.01, 1]

# make prediction at different weeks
for week in ('82.0', '86.0', '90.0'):

    print('Week: ', week)

    # read dataset
    df = pd.read_csv('../outputs/prediction_dataset/data at week {}.csv'.format(week))
    # randomize rows
    df = df.sample(frac=1, random_state=10)

    # print('Linear regression models:')
    # E.evaluate_linear_regression(data=df, feature_names=feature_names, outcome_name=OUTCOME,
    #                              poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS,
    #                              if_standardize=IF_STANDARDIZED, penalty=PENALTY)

    print('Neural network models:')
    E.evaluate_neural_network(data=df, feature_names=feature_names, outcome_name=OUTCOME,
                              list_of_alphas=[0.0001],
                              list_of_num_fs_wanted=[10],
                              feature_selection_method='pi',
                              cv_fold=10, outcome_deci=4, week=week,
                              if_standardize=IF_STANDARDIZED, num_of_neurons=None, save_to_file=True)

    print('\n')
