from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

# use the entire dataset to predict the probability of surge using logistic regression and neural network

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
y_name = 'If hospitalization threshold passed'

######################
# LOGISTIC REGRESSION #
######################
df_lr = Dataframe(df=df, features=features, y_name=y_name)
# pre-processing
df_lr.preprocess(if_standardize=True, degree_of_polynomial=2)
# feature selection
df_lr.feature_selection(method='rfe',
                        # liblinear is a good choice for small dataset, sag’ and ‘saga’ are faster for large ones.
                        estimator=linear_model.LogisticRegression(penalty="l1", C=0.1, solver='liblinear'),
                        num_fs_wanted=15)
print('selected features:', df_lr.selected_features)
print('number of selected features', len(df_lr.selected_features))

# construct model cohort
multi_log_model = MultiLogisticReg(df=df_lr.df, features=df_lr.selected_features, y_name=df_lr.y_name)
multi_log_model.run_many(num_bootstraps=10)

##################
# NEURAL NETWORK #
##################


