import pandas as pd

import covid_prediction.evaluate_predictive_models as E

root_path = '/Users/shiyingyou/PycharmProjects/PredictingCOVID19Surge'

# read dataset
df = pd.read_csv('{}/outputs/prediction_datasets/data at week 4.csv'.format(root_path))
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
cv_fold = 10         # num of splits for cross validation
feature_selection = 'pi'
list_of_num_features_wanted = [10, 20, 30, 40]
list_of_alphas = [0.0001, 0.001, 0.01, 1]

# make prediction at different weeks
for week in ('4', '8', '12'):

    print('Week: ', week)

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/data at week {}.csv'.format(root_path, week))
    # randomize rows
    df = df.sample(frac=1, random_state=10)

    print('Linear regression models:')
    E.evaluate_linear_regression(data=df, feature_names=feature_names, outcome_name=OUTCOME,
                                 cv_fold=cv_fold, outcome_deci=decimal, week=week,
                                 list_of_penalties=[['l1', 1.0], ['l2', 1.0], ['None', 'None']],
                                 list_of_poly_degrees=[1, 2],
                                 list_of_num_fs_wanted=[10, 20, 30, 40],
                                 feature_selection_method='rfe',      # pi/rfe, did not include lasso
                                 if_standardize=IF_STANDARDIZED, save_to_file=True)

    print('Neural network models:')
    E.evaluate_neural_network(data=df, feature_names=feature_names, outcome_name=OUTCOME,
                              list_of_alphas=[0.0001, 0.001],
                              list_of_num_fs_wanted=[10, 20],
                              feature_selection_method='pi',
                              cv_fold=cv_fold, outcome_deci=decimal, week=week,
                              if_standardize=IF_STANDARDIZED, num_of_neurons=None, save_to_file=True)

    print('\n')
