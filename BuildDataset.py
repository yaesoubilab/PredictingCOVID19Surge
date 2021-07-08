from covid_prediction.feature_engineering import *
from definitions import *

ICU_CAPACITY = 10.34  # beds per 100,000 population
ICU_CAPACITY_RATE = 0.0001


# our goal is to use the data during [0, D.CALIB_PERIOD]
# to predict if the ICU capacity would be surpassed during [D.CALIB_PERIOD, D.SIM_DURATION]

# all trajectories are located in 'outputs/trajectories'
# column 'Observation Time' represent year and column 'Observation Period' represent the period (week)
# ICU occupancy over time is in column 'Obs: Hospitalization rate'

# to determine if surge has occurred for a trajectory, we check if the
# value of column 'Obs: Hospitalization rate' passes ICU_CAPACITY during [D.CALIB_PERIOD, D.SIM_DURATION].

# one of the important things we need to decide is what predictors to use.
# for now, let's use these:
#   1) 'Obs: Cumulative vaccination' at year D.CALIB_PERIOD
#   2) 'Obs: Incidence' at week D.CALIB_PERIOD * 52


# read dataset
trajs = DataEngineering(directory_name='outputs/trajectories',
                        calib_period=CALIB_PERIOD,
                        proj_period=PROJ_PERIOD,
                        icu_capacity_rate=ICU_CAPACITY_RATE)
trajs.read_datasets()

# create new dataset based on raw data
df = trajs.pre_processing()

# save new dataset to file
df.to_csv('outputs/prediction_dataset/cleaned_data.csv', index=False)
