# Predicting COVID-19 Surge in Fall and Spring 2022

## Dependencies
Packages: `deampy (1.0.17)`, `apacepy (1.0.5)`, `pydotplus (2.0.2)`, and `imblearn (0.0)`.

## To create decision rules:

1. Calibrate the simulation model by running the script [calibrate.py](calibrate.py). 
   This script will identify a set of simulated trajectories that can be used 
   to develop and validate the decision rules.
2. Build all datasets for developing and validating the decision rules by running
   [build_all_datasets.py](build_all_datasets.py). This scrip uses the simulated trajectories 
   identified by the calibration procedure to create the dataset needed 
   to develop and validate the decision rules. 
3. Build and validate decision trees by running [build_and_validate_decision_trees.py](build_and_validate_decision_trees.py).
   1. The decision rules will be stored under 
      [outputs/figures/trees_4_weeks](outputs/figures/trees_4_weeks) and 
      [outputs/figures/trees_8_weeks](outputs/figures/trees_8_weeks).  
   2. The performance of decision rules under different scenarios will be stored under
      [outputs/prediction_summary_4_weeks/dec_tree/summary.csv](outputs/prediction_summary_4_weeks/dec_tree/summary.csv) and
      [outputs/prediction_summary_8_weeks/dec_tree/summary.csv](outputs/prediction_summary_8_weeks/dec_tree/summary.csv).
4. If you would like to create a pruner decision trees, use [build_a_decision_tree.py](build_a_decision_tree.py)
   with a higher value of CCP_ALPHA.

## Other functions:
- Run [simulate_many.py](simulate_many.py) to generate simulated trajectories, 
  which will be stored in
  [outputs/trajectories](outputs/trajectories). 
  Figure visualizing these trajectories will be stored under [outputs/figures](outputs/figures).