# Predicting COVID-19 Surge in Fall 2021

## Dependencies
Add the following libraries to the project Content Root.
1. [APACE](https://github.com/yaesoubilab/APACE)
2. [SimPy](https://github.com/yaesoubilab/SimPy)

## Workflow

1. Run [SimulateMany.py](SimulateMany.py) to generate simulated trajectories, which will be stored in
[outputs/trajectories](outputs/trajectories).
2. Run [BuildDataset.py](BuildDataset.py) to use the simulated trajectories create the dataset needed 
   to develop machine-learning models (saves it under [outputs](outputs)).
3. Run [Predict.py](Predict.py) to develop machine learning models to predict if surge would occur in Fall 2021.
