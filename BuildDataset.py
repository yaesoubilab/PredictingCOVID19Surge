from covid_prediction.feature_engineering import *
from definitions import ROOT_DIR, get_dataset_labels, get_outcome_label, \
    FEASIBILITY_PERIOD, N_NOVEL_INCD

HOSP_OCCU_THRESHOLDS = (10, 15, 20)  # per 100,000 population
TIME_OF_FALL = FEASIBILITY_PERIOD


def build_dataset(week_of_prediction_in_fall,
                  pred_period,
                  hosp_thresholds,
                  survey_size_novel_inf,
                  report_corr=True):
    """ create the dataset needed to develop the predictive models
    :param week_of_prediction_in_fall: (int) a positive int for number of weeks into fall and
                                             a negative int for number of weeks before the peak
    :param pred_period: (tuple) (y0, y1) time (in year) when the prediction period starts and ends;
        it is assumed that prediction is made at time y0 for an outcome observed during (y0, y1).
    :param hosp_thresholds: (list) of thresholds of hospitalization capacity
    :param survey_size_novel_inf: (int) survey size of novel infection surveillance
    :param report_corr: (bool) whether to report correlations between features and outcomes
    """

    # read dataset
    feature_engineer = FeatureEngineering(
        dir_of_trajs='outputs/trajectories',
        week_of_prediction_in_fall=week_of_prediction_in_fall,
        pred_period=pred_period,
        hosp_thresholds=hosp_thresholds)

    # error model for novel variant surveillance
    err_novel_incd = ErrorModel(survey_size=survey_size_novel_inf)

    # find output file name
    label = get_dataset_labels(week=week_of_prediction_in_fall, survey_size=survey_size_novel_inf)
    output_file = 'data-{}.csv'.format(label)

    # create new dataset based on raw data
    feature_engineer.pre_process(
        # information for incidence and prevalence features can be provided as
        #   'name of the feature' to calculate the recording at prediction time
        # or
        #   (feature's name, 100, ('ave', 2), ('slope', 4))
        # to multiply the column values as needed,
        # to report the recording at prediction time, and
        # to calculate the average and slope of observations during past weeks
        info_of_incd_fs=[
            ('Obs: Incidence rate', 100000, ('ave', 2), ('slope', 4)),
            ('Obs: New hospitalization rate', 100000, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations that are vaccinated', ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence due to novel variant', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to novel variant', ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to Novel-V', ('ave', 2), ('slope', 4)),
        ],
        info_of_prev_fs=[
            ('Obs: Hospital occupancy rate', 100000),
            ('Obs: Cumulative hospitalization rate', 100000),
            ('Obs: Cumulative vaccination rate', 1),
            ('Obs: Prevalence susceptible', 1)
        ],
        info_of_parameter_fs=[
            'R0',
            'Duration of infectiousness-dominant',
            'Prob novel strain params-0',
            'Prob novel strain params-1',
            'Prob novel strain params-2',
            'Ratio infectiousness duration of novel to dominant',
            'Duration of vaccine immunity',
            'Ratio of duration of immunity from infection+vaccination to infection',
            'Vaccine effectiveness against infection-0',
            'Vaccine effectiveness against infection-1',
            'Ratio transmissibility of novel to dominant',
            'Vaccine effectiveness in reducing infectiousness-0',
            'Vaccine effectiveness in reducing infectiousness-1',
            'Ratio prob of hospitalization of novel to dominant',
            'Vaccine effectiveness against hospitalization-0',
            'Vaccine effectiveness against hospitalization-1',
            'Prob Hosp for 18-29',
            'Relative prob hosp by age-0',
            'Relative prob hosp by age-1',
            'Relative prob hosp by age-2',
            'Relative prob hosp by age-4',
            'Relative prob hosp by age-5',
            'Relative prob hosp by age-6',
            'Relative prob hosp by age-7',
            'Duration of E-0',
            'Duration of E-1',
            'Duration of E-2',
            'Duration of E-3',
            'Duration of I-0',
            'Duration of I-1',
            'Duration of I-2',
            'Duration of I-3',
            'Duration of Hosp-0',
            'Duration of Hosp-1',
            'Duration of Hosp-2',
            'Duration of Hosp-3',
            'Duration of R-0',
            'Duration of R-1',
            'Duration of R-2',
            'Duration of R-3',
            'Vaccination rate params-0',
            'Vaccination rate params-1',
            'Vaccination rate params-3',
            'Vaccination rate t_min by age-1',
            'Vaccination rate t_min by age-2',
            'Vaccination rate t_min by age-3',
            'Vaccination rate t_min by age-4',
            'Vaccination rate t_min by age-5',
            'Vaccination rate t_min by age-6',
            'Vaccination rate t_min by age-7',
            'Y1 thresholds-0',
            'Y1 thresholds-1',
            'Y1 Maximum hosp occupancy',
            'Y1 Max effectiveness of control measures'
        ],
        output_file=output_file,
        report_corr=report_corr)


def build_and_combine_datasets(weeks_in_fall, weeks_to_predict, hosp_occu_thresholds, survey_size_novel_inf):
    """
    :param weeks_in_fall: (list) of weeks into fall where feature values should be recorded
    :param weeks_to_predict: (int) number of weeks in future when outcomes should be predicted
    :param hosp_occu_thresholds: (list) of thresholds for hospital occupancy
    :param survey_size_novel_inf: (int) survey size of novel infection surveillance
    """

    # datasets for predicting whether hospitalization capacities would surpass withing 4 weeks
    for week_in_fall in weeks_in_fall:

        # find the time (year) for which the prediction should be made
        time_of_prediction = TIME_OF_FALL + week_in_fall / 52

        # build the dataset
        build_dataset(week_of_prediction_in_fall=week_in_fall,
                      pred_period=(time_of_prediction, time_of_prediction + weeks_to_predict / 52),
                      hosp_thresholds=hosp_occu_thresholds,
                      survey_size_novel_inf=survey_size_novel_inf,
                      report_corr=False)

    # merge the data collected at different weeks to from a
    # single dataset for training the model
    dataframes = []
    prefix = '/outputs/prediction_datasets/week_into_fall/'
    for w in weeks_in_fall:
        label = get_dataset_labels(week=w, survey_size=survey_size_novel_inf)
        dataframes.append(pd.read_csv(
            ROOT_DIR + prefix + 'data-{}.csv'.format(label)))
    dataset = pd.concat(dataframes)
    dataset.to_csv(ROOT_DIR + prefix + 'combined data-sample size {}.csv'.format(survey_size_novel_inf),
                   index=False)

    # report the % of observations where hospital occupancy threshold passes
    for t in hosp_occu_thresholds:
        print('% observations where threshold {} is passed:'.format(t),
              round(100 * (1 - dataset[get_outcome_label(threshold=t)].mean()), 1))

    # report correlation
    outcomes = [get_outcome_label(threshold=t) for t in hosp_occu_thresholds]
    report_corrs(df=dataset,
                 outcomes=outcomes,
                 csv_file_name=ROOT_DIR + prefix + 'corr-sample size {}.csv'.format(survey_size_novel_inf))


if __name__ == "__main__":

    # build datasets for prediction at certain weeks:
    # fall/winter start in week 78 and end on 117
    build_and_combine_datasets(weeks_in_fall=(8, 12, 16, 20, 24, 28, 32),
                               weeks_to_predict=4,
                               hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
                               survey_size_novel_inf=N_NOVEL_INCD)

