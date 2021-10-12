import apace.Calibration as calib
from BuildTrainingDataset import build_and_combine_datasets
from SimulateMany import simulate
from definitions import FEASIBILITY_PERIOD, N_SIM_VALIDATION, N_SIM_TRAINING, \
    N_NOVEL_INCD, SCENARIOS, WEEKS_IN_FALL, HOSP_OCCU_THRESHOLDS


def build_validation_datasets():

    # ---- build the dataset with smaller survey size ----
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['smaller survey'],
        time_of_fall=FEASIBILITY_PERIOD,
        weeks_in_fall=WEEKS_IN_FALL,
        weeks_to_predict=4,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=200
    )

    # ---- build the dataset for the base scenario ----
    # get the seeds and probability weights
    seeds = calib.get_seeds_with_non_zero_prob('outputs/summary/calibration_summary.csv')
    if len(seeds) <= N_SIM_TRAINING + N_SIM_VALIDATION:
        print('Warning: there is overlap between simulation trajectories used for training and validation.')
    # simulate
    simulate(n=N_SIM_VALIDATION,
             seeds=seeds[-N_SIM_VALIDATION:],
             sample_seeds_by_weights=False)
    # build the dataset
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['base'],
        time_of_fall=FEASIBILITY_PERIOD,
        weeks_in_fall=WEEKS_IN_FALL,
        weeks_to_predict=4,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD
    )

    # ---- build the dataset with no novel variant ----

    # ---- build the dataset with no mitigating strategies ----


if __name__ == '__main__':
    build_validation_datasets()
