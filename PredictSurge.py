import pandas as pd

import covid_prediction.evaluate_predictive_models as E

features = ['Obs: New hospitalization rate',
            'Obs: Cumulative vaccination rate', 'Obs: Cumulative hospitalization rate',
            'R0s-0']
OUTCOME = 'If hospitalization threshold passed'
NUM_OF_BOOTSTRAPS = 100
POLY_DEGREE = 1
IF_STANDARDIZED = True
PENALTY = 'l2' # 'l1', 'l2', or 'none'

# read dataset
df = pd.read_csv('outputs/prediction_dataset/data at week 78.0.csv')

E.evaluate_logistic(data=df, feature_names=features[0:1], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)

E.evaluate_logistic(data=df, feature_names=features[1:2], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)

E.evaluate_logistic(data=df, feature_names=features[2:3], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)

E.evaluate_logistic(data=df, feature_names=[features[0], features[2]], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)

E.evaluate_logistic(data=df, feature_names=features[0:4], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)


# E.evaluate_tree(data=df, feature_names=features[0:4], outcome_name=OUTCOME,
#                 n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED)
