import pandas as pd

import covid_prediction.evaluate_predictive_models as E
from definitions import ROOT_DIR

POLY_DEGREE = 1
IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi' # could be 'rfe', 'lasso', or 'pi'
CV_FOLD = 10         # num of splits for cross validation

# read dataset
df = pd.read_csv(ROOT_DIR+'/outputs/prediction_datasets/data at week 4.csv')
# feature names (all columns are considered)
feature_names = df.columns.tolist()
feature_names.remove('Maximum hospitalization rate')
feature_names.remove('If hospitalization threshold passed')

# make prediction at different weeks
for week in ('4', '8', '12'):

    print('Week: ', week)

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/data at week {}.csv'.format(ROOT_DIR, week))
    # randomize rows
    df = df.sample(frac=1, random_state=1)

    # print('Linear regression models:')
    # E.evaluate_linear_regression(data=df, feature_names=feature_names, outcome_name=OUTCOME,
    #                              cv_fold=CV_FOLD, outcome_deci=DECIMAL, week=week,
    #                              list_of_penalties=[['l1', 1.0], ['l2', 1.0], ['None', 'None']],
    #                              list_of_poly_degrees=[1, 2],
    #                              list_of_num_fs_wanted=[10, 20, 30, 40],
    #                              feature_selection_method='rfe',  # pi/rfe, did not include lasso
    #                              if_standardize=IF_STANDARDIZED, save_to_file=True)

    print('Neural network models:')
    E.evaluate_neural_network(data=df, feature_names=feature_names,
                              outcome_name='Maximum hospitalization rate',
                              list_of_num_fs_wanted=[10, 20],
                              list_of_alphas=[0.0001, 0.001],
                              list_of_n_neurons=[10, 20],
                              feature_selection_method=FEATURE_SELECTION,
                              cv_fold=CV_FOLD, if_standardize=IF_STANDARDIZED,
                              save_to_file='NN eval-{}.csv'.format(week))

    print('\n')
