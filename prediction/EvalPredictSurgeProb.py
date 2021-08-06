import pandas as pd

import covid_prediction.evaluate_predictive_models as E

features = ['Obs: New hospitalization rate',
            'Obs: Cumulative vaccination rate',
            'Obs: Cumulative hospitalization rate',
            'R0',
            'Duration of infectiousness-dominant',
            'Prob Hosp for 18-29',
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
            'Vaccine effectiveness against infection with novel']
OUTCOME = 'If hospitalization threshold passed'
NUM_OF_BOOTSTRAPS = 100
POLY_DEGREE = 2
C = 0.1
IF_STANDARDIZED = True
PENALTY = 'l2' # 'l1', 'l2', or 'none'

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week 78.0.csv')

# model = LogisticReg(df=df, features=features, y_name=OUTCOME)
# model.run(random_state=RandomState(0), degree_of_polynomial=POLY_DEGREE, C=C)
# print(model.coeffs)
# print(model.intercept)

E.evaluate_logistic(data=df, feature_names=features[0:1], outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, C=C, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)

features = ['Obs: New hospitalization rate',
            'Obs: Cumulative vaccination rate',
            'Obs: Cumulative hospitalization rate',
            'R0',
            'Duration of infectiousness-dominant',
            'Duration of vaccine immunity',
            'Vaccine effectiveness against infection with novel']
E.evaluate_logistic(data=df, feature_names=features, outcome_name=OUTCOME,
                    poly_degree=POLY_DEGREE, C=C, n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED,
                    penalty=PENALTY)



# E.evaluate_tree(data=df, feature_names=features[0:4], outcome_name=OUTCOME,
#                 n_bootstraps=NUM_OF_BOOTSTRAPS, if_standardize=IF_STANDARDIZED)
