# Predicting COVID-19 Surge in Fall and Spring 2022

## Dependencies
Packages: `deampy`, `apacepy`, `pydotplus`, and `imblearn`.

## To create decision rules:

1. Calibrate the simulation model by running the script [calibrate.py](calibrate.py). 
   This script will identify a set of simulated trajectories that can be used 
   to develop and validate the decision rules.
2. Build all datasets for developing and validating the decision rules by running
   [build_all_datasets.py](build_all_datasets.py). This scrip uses the simulated trajectories 
   identified by the calibration procedure to create the dataset needed 
   to develop and validate the decision rules. 


## Other functions:
- Run [simulate_many.py](simulate_many.py) to generate simulated trajectories, 
  which will be stored in
  [outputs/trajectories](outputs/trajectories).