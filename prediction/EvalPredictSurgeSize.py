import pandas as pd

import covid_prediction.evaluate_predictive_models as E

features = ['Obs: New hospitalization rate',
            # 'Obs: % of incidence due to Novel',
            # 'R0s-0',
            'Obs: Cumulative vaccination rate',
            'Obs: Cumulative hospitalization rate'
            ]
# TODO: I commented two features out because I could not find them in the dataset

OUTCOME = 'Maximum hospitalization rate'
NUM_OF_BOOTSTRAPS = 10
POLY_DEGREE = 1
IF_STANDARDIZED = True
PENALTY = 'l2' # 'l1', 'l2', or 'none'

# hyper-parameters for NN
ACTIVATION = 'logistic'
SOLVER = 'sgd'
MAX_ITER = 1000
ALPHA = 0.0001


# make prediction at different weeks
for week in ('82.0', '86.0', '90.0'):

    print('Week: ', week)

    # read dataset
    # df = pd.read_csv('outputs/prediction_dataset/data at week {}.csv'.format(week))
    df = pd.read_csv('/Users/shiyingyou/PycharmProjects/PredictingCOVID19Surge/outputs/prediction_dataset/data at week {}.csv'.format(week))

    print('Linear regression models:')
    E.evaluate_linear_regression(data=df, feature_names=features, outcome_name=OUTCOME,
                                 poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                                 penalty=PENALTY)

    print('Neural network models:')
    E.evaluate_neural_network(data=df, feature_names=features, outcome_name=OUTCOME,
                              n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                              activation=ACTIVATION, solver=SOLVER, alpha=ALPHA, max_iter=MAX_ITER)

    print('\n')
