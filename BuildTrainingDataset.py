from covid_prediction.pre_process import build_and_combine_datasets
from definitions import FEASIBILITY_PERIOD, N_NOVEL_INCD

WEEKS_IN_FALL = (8, 12, 16, 20, 24, 28, 32)
HOSP_OCCU_THRESHOLDS = (10, 15, 20)  # per 100,000 population
TIME_OF_FALL = FEASIBILITY_PERIOD

if __name__ == "__main__":

    data_type = 'data-training'
    # data_type = 'no control measure'
    # data_type = 'no novel variant'
    # data_type = 'survey size 200'

    # build datasets for prediction at certain weeks:
    # fall/winter start in week 78 and end on 117
    build_and_combine_datasets(
        name_of_dataset=data_type,
        time_of_fall=TIME_OF_FALL,
        weeks_in_fall=WEEKS_IN_FALL,
        weeks_to_predict=4,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=N_NOVEL_INCD)

