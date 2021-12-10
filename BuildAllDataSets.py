from BuildTrainingDataset import build_training_dataset
from BuildValidationDatasets import build_validation_datasets
from SimulateMany import simulate
from definitions import N_SIM_TRAINING

if __name__ == "__main__":

    simulate(n=N_SIM_TRAINING,
             n_to_display=min(100, N_SIM_TRAINING),
             calibrated=True,
             novel_variant_will_emerge=True,
             mitigating_strategies_on=True)

    for w in (4, 8):

        build_training_dataset(weeks_to_predict=w)
        build_validation_datasets(weeks_to_predict=w)
