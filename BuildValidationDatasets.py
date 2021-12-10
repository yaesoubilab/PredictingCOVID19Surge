import sys
import warnings

import apace.Calibration as calib
from BuildTrainingDataset import build_and_combine_datasets
from SimulateMany import simulate
from definitions import ROOT_DIR, SIM_DURATION, N_SIM_VALIDATION, N_SIM_TRAINING, \
    N_NOVEL_INCD, SCENARIOS, FIRST_WEEK_OF_WINTER, HOSP_OCCU_THRESHOLDS, WEEKS_TO_PREDICT, SMALLER_N_NOVEL_INCD


def build_validation_datasets():

    warnings.filterwarnings("ignore")
    sys.stdout = open(
        ROOT_DIR+'/outputs/prediction_datasets/summary_validating_set.txt', 'w')

    # ---- build the dataset for the base scenario ----
    # get the seeds and probability weights
    seeds = calib.get_seeds_with_non_zero_prob(
        filename='outputs/summary/calibration_summary.csv',
        random_state=0)
    if len(seeds) <= N_SIM_TRAINING + N_SIM_VALIDATION:
        print('** Warning **: there is overlap between simulation trajectories used for training and validation.')

    # simulate a new set of trajectories for validation
    simulate(n=N_SIM_VALIDATION,
             seeds=seeds[-N_SIM_VALIDATION:],
             n_to_display=N_SIM_VALIDATION,
             sample_seeds_by_weights=False,
             print_summary_stats=False,
             folder_to_save_plots=ROOT_DIR + '/outputs/figures/scenarios/base')

    # ---- build the dataset for the base scenario ----
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['base'],
        first_week_of_winter=FIRST_WEEK_OF_WINTER,
        last_week_of_winter=int(SIM_DURATION*52),
        weeks_to_predict=WEEKS_TO_PREDICT,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD
    )

    # ---- build the dataset with smaller survey size ----
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['smaller survey'],
        first_week_of_winter=FIRST_WEEK_OF_WINTER,
        last_week_of_winter=int(SIM_DURATION*52),
        weeks_to_predict=WEEKS_TO_PREDICT,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=SMALLER_N_NOVEL_INCD,
    )

    # ---- build the dataset with no novel variant ----
    # simulate
    simulate(n=N_SIM_VALIDATION,
             n_to_display=N_SIM_VALIDATION,
             sample_seeds_by_weights=False,
             novel_variant_will_emerge=False,
             print_summary_stats=False,
             folder_to_save_plots=ROOT_DIR + '/outputs/figures/scenarios/no_novel_variant')
    # build the dataset
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['no novel variant'],
        first_week_of_winter=FIRST_WEEK_OF_WINTER,
        last_week_of_winter=int(SIM_DURATION*52),
        weeks_to_predict=WEEKS_TO_PREDICT,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD
    )

    # ---- build the dataset with no mitigating strategies ----
    # simulate
    simulate(n=N_SIM_VALIDATION,
             n_to_display=N_SIM_VALIDATION,
             sample_seeds_by_weights=False,
             mitigating_strategies_on=False,
             print_summary_stats=False,
             folder_to_save_plots=ROOT_DIR + '/outputs/figures/scenarios/no_mitigation')
    # build the dataset
    build_and_combine_datasets(
        name_of_dataset='data-validating ' + SCENARIOS['no control measure'],
        first_week_of_winter=FIRST_WEEK_OF_WINTER,
        last_week_of_winter=int(SIM_DURATION*52),
        weeks_to_predict=WEEKS_TO_PREDICT,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD
    )

    sys.stdout.close()


if __name__ == '__main__':
    build_validation_datasets()
