ICU_CAPACITY = 10.34 # beds per 100,000 population

# our goal is to use the data during [0, D.CALIB_PERIOD]
# to predict if the ICU capacity would be surpassed during [D.CALIB_PERIOD, D.SIM_DURATION]

# all trajectories are located in 'outputs/trajectories'
# column 'Observation Time' represent *year* and column 'Observation Period' represent the period (*week*)
# ICU occupancy over time is in column 'Obs: # in ICU'

# to determine if surge has occurred for a trajectory, we check if the
# value of column 'Obs: # in ICU' passes ICU_CAPACITY during [D.CALIB_PERIOD, D.SIM_DURATION].

# one of the important things we need to decide is what predictors to use.
# for now, let's use these these:
#   1) 'Obs: Vaccinated' at year D.CALIB_PERIOD
#   2) 'Obs: Incidence' at week D.CALIB_PERIOD * 52
