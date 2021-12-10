import sys

from SimPy.InOutFunctions import make_directory
from covid_prediction.pre_process import build_and_combine_datasets
from covid_prediction.summary_of_trajs import report_traj_summary
from definitions import SIM_DURATION, N_NOVEL_INCD, FIRST_WEEK_OF_WINTER, \
    HOSP_OCCU_THRESHOLDS, ROOT_DIR, WEEKS_TO_PREDICT


def build_training_dataset():

    directory = ROOT_DIR + '/outputs/prediction_datasets_{}_weeks/'.format(WEEKS_TO_PREDICT)
    make_directory(filename=directory)
    sys.stdout = open(
        directory + 'summary_training_set.txt', 'w')

    # the report the summary of training trajectories
    report_traj_summary(hosp_occ_thresholds=HOSP_OCCU_THRESHOLDS,
                        perc_novel_thresholds=[0.05, 0.95])

    # build datasets for prediction at certain weeks:
    # fall/winter start in week 78 and end on 117
    build_and_combine_datasets(
        name_of_dataset='data-training',
        first_week_of_winter=FIRST_WEEK_OF_WINTER,
        last_week_of_winter=int(SIM_DURATION * 52),
        weeks_to_predict=WEEKS_TO_PREDICT,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD)

    sys.stdout.close()


if __name__ == "__main__":

    build_training_dataset()
