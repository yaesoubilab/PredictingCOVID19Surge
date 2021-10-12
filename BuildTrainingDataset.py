from covid_prediction.pre_process import build_and_combine_datasets
from definitions import FEASIBILITY_PERIOD, N_NOVEL_INCD, WEEKS_IN_FALL, HOSP_OCCU_THRESHOLDS


if __name__ == "__main__":

    # build datasets for prediction at certain weeks:
    # fall/winter start in week 78 and end on 117
    build_and_combine_datasets(
        name_of_dataset='data-training',
        time_of_fall=FEASIBILITY_PERIOD,
        weeks_in_fall=WEEKS_IN_FALL,
        weeks_to_predict=4,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD)

