from covid_prediction.feature_engineering import *
from definitions import *

HOSPITALIZATION_THRESHOLD = 0.0001  # per 100,000 population


# our goal is to use the data during [0, D.CALIB_PERIOD]
# to predict 1) if the hospitalization would be surpassed and and 2) the maximum hospitalization rate
# during [D.CALIB_PERIOD, D.SIM_DURATION]

# all trajectories are located in 'outputs/trajectories'
# column 'Observation Time' represent year and column 'Observation Period' represent the period (week)
# Hospitalization over time is in column 'Obs: Hospitalization rate'

# to determine if surge has occurred for a trajectory, we check if the
# value of column 'Obs: Hospitalization rate' passes
# HOSPITALIZATION_THRESHOLD during [D.CALIB_PERIOD, D.SIM_DURATION].

# one of the important things we need to decide is what predictors to use.
# for now, let's use these:
#   1) 'Obs: Cumulative vaccination' at year D.CALIB_PERIOD
#   2) 'Obs: Incidence' at week D.CALIB_PERIOD * 52


# read dataset
feature_engineer = FeatureEngineering(
    directory_name='outputs/trajectories',
    calib_period=CALIB_PERIOD,
    proj_period=PROJ_PERIOD,
    hosp_threshold=HOSPITALIZATION_THRESHOLD)

# create new dataset based on raw data
df = feature_engineer.pre_process(
    names_of_incidence_features=['Obs: Incidence'],
    names_of_prevalence_features=['Obs: Cumulative vaccination'],
    file_name='outputs/prediction_dataset/cleaned_data.csv')
