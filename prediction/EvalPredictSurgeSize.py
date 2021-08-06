import pandas as pd

import covid_prediction.evaluate_predictive_models as E

features = ['Obs: New hospitalization rate',
            'Obs: % of incidence due to Novel',
            'Obs: Cumulative vaccination rate',
            'Obs: Cumulative hospitalization rate',
            'R0s-0']
OUTCOME = 'Maximum hospitalization rate'
NUM_OF_BOOTSTRAPS = 100
POLY_DEGREE = 1
IF_STANDARDIZED = True
PENALTY = 'l2' # 'l1', 'l2', or 'none'


# make prediction at different weeks
for week in ('78.0', '82.0', '86.0'):

    print('Week: ', week)

    # read dataset
    df = pd.read_csv('outputs/prediction_dataset/data at week {}.csv'.format(week))

    E.evaluate_linear_regression(data=df, feature_names=[features[0], features[2], features[3]], outcome_name=OUTCOME,
                                 poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                                 penalty=PENALTY)

    E.evaluate_linear_regression(data=df, feature_names=features, outcome_name=OUTCOME,
                                 poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                                 penalty=PENALTY)
