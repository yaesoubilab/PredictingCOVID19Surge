import pandas as pd

import covid_prediction.cross_validation as CV
from definitions import ROOT_DIR

IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi' # could be 'rfe', 'lasso', or 'pi'
CV_FOLD = 10         # num of splits for cross validation


def get_nue_net_best_performance(week, feature_names):

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/data at week {}.csv'.format(ROOT_DIR, week))

    # use all features if no feature name is provided
    if feature_names is None:
        # feature names (all columns are considered)
        feature_names = df.columns.tolist()
        feature_names.remove('Maximum hospitalization rate')
        feature_names.remove('If hospitalization threshold passed')

    # randomize rows
    df = df.sample(frac=1, random_state=1)

    # find the best specification
    cv = CV.NeuNetSepecOptimizer(data=df, feature_names=feature_names,
                                 outcome_name='Maximum hospitalization rate',
                                 list_of_num_fs_wanted=[10, 20],
                                 list_of_alphas=[0.0001, 0.001],
                                 list_of_n_neurons=[10, 20],
                                 feature_selection_method=FEATURE_SELECTION,
                                 cv_fold=CV_FOLD, if_standardize=IF_STANDARDIZED)

    best_spec = cv.find_best_spec(run_in_parallel=False,
                                  save_to_file=ROOT_DIR + '/outputs/prediction_summary/NN eval-{}.csv'.format(week))
    return best_spec


# make prediction at different weeks
for week in ('8', '12'):

    print('Week: ', week)

    best_spec = get_nue_net_best_performance()

    print('\n')
