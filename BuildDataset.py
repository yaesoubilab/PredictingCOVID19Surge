from covid_prediction.feature_engineering import *
from definitions import ROOT_DIR, get_dataset_labels, FEASIBILITY_PERIOD, SIM_DURATION, OUTCOMES_IN_DATASET


HOSPITALIZATION_THRESHOLD = 10.3/100000  # per 100,000 population
TIME_OF_FALL = FEASIBILITY_PERIOD

# survey sizes
N_NOVEL_INCD = 1521
N_PREV_SUSC = 481
N_HOSP_UNVAC = 864


def build_dataset(week_of_prediction_in_fall, pred_period, hosp_threshold,
                  noise_coeff=None, bias_delay=None, report_corr=True):
    """ create the dataset needed to develop the predictive models
    :param week_of_prediction_in_fall: (int) a positive int for number of weeks into fall and
                                             a negative int for number of weeks before the peak
    :param pred_period: (tuple) (y0, y1) time (in year) when the prediction period starts and ends;
        it is assumed that prediction is made at time y0 for an outcome observed during (y0, y1).
    :param hosp_threshold: threshold of hospitalization capacity
    :param noise_coeff: (None or int) if None, the noise model is not added, otherwise, the noise model is
        added with survey size multiplied by add_noise.
    :param bias_delay: (None or int) delay (in weeks) of observing the true value
    :param report_corr: (bool) whether to report correlations between features and outcomes
    """
    # read dataset
    feature_engineer = FeatureEngineering(
        dir_of_trajs='outputs/trajectories',
        week_of_prediction_in_fall=week_of_prediction_in_fall,
        pred_period=pred_period,
        hosp_threshold=hosp_threshold)

    # error models
    err_novel_incd = None   # error model for % of incidence with novel variant
    err_prev_susc = None    # error model for prevalence susceptible
    err_hosp_vacc = None  # error model for the % of hospitalized patients who are vaccinated
    # if bias needs to be added
    if bias_delay is not None:
        # and if noise needs to be added
        if noise_coeff is not None:
            err_novel_incd = ErrorModel(
                survey_size=N_NOVEL_INCD * noise_coeff, bias_delay=bias_delay)
            err_prev_susc = ErrorModel(
                survey_size=N_PREV_SUSC * noise_coeff, bias_delay=bias_delay)
            err_hosp_vacc = ErrorModel(
                survey_size=N_HOSP_UNVAC, bias_delay=bias_delay)
    else:  # no bias (only noise)
        # if noise needs to be added
        if noise_coeff is not None:
            err_novel_incd = ErrorModel(survey_size=N_NOVEL_INCD * noise_coeff)
            err_prev_susc = ErrorModel(survey_size=N_PREV_SUSC * noise_coeff)
            err_hosp_vacc = None  # no error

    # find output file name
    label = get_dataset_labels(
        week=week_of_prediction_in_fall, noise_coeff=noise_coeff, bias_delay=bias_delay)
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
            ('Obs: % of new hospitalizations that are vaccinated', err_hosp_vacc, ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence due to novel variant', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to novel variant', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to Novel-V', err_hosp_vacc, ('ave', 2), ('slope', 4)),
        ],
        info_of_prev_fs=[
            ('Obs: Hospital occupancy rate', 100000),
            ('Obs: Cumulative hospitalization rate', 100000),
            ('Obs: Cumulative vaccination rate', 1),
            ('Obs: Prevalence susceptible', err_prev_susc)],
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


def build_and_combine_datasets(weeks_in_fall, weeks_to_predict, hosp_threshold):

    # datasets for predicting whether hospitalization capacities would surpass withing 4 weeks
    for week_in_fall in weeks_in_fall:
        time_of_prediction = TIME_OF_FALL + week_in_fall / 52
        build_dataset(week_of_prediction_in_fall=week_in_fall,
                      pred_period=(time_of_prediction, time_of_prediction + weeks_to_predict / 52),
                      hosp_threshold=hosp_threshold,
                      report_corr=False)

    # merge the data collected at different weeks to from a
    # single dataset for training the model
    dataframes = []
    for w in weeks_in_fall:
        dataframes.append(pd.read_csv(
            ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/data-wk {}.csv'.format(w)))
    dataset = pd.concat(dataframes)
    dataset.to_csv(ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/combined_data.csv',
                   index=False)

    print('% observations where threshold is not passed:',
          round(dataset[OUTCOMES_IN_DATASET[1]].mean()*100, 1))

    # report correlation
    report_corrs(df=dataset, outcomes=OUTCOMES_IN_DATASET,
                 csv_file_name=ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/corr.csv')


if __name__ == "__main__":

    # build datasets for prediction at certain weeks:
    # fall/winter start in week 78 and end on 117
    build_and_combine_datasets(weeks_in_fall=(8, 12, 16, 20, 24, 28, 32),
                               weeks_to_predict=4,
                               hosp_threshold=HOSPITALIZATION_THRESHOLD)

    # datasets for prediction at cetain weeks until peak
    datasets_for_pred_negative_weeks = False
    if datasets_for_pred_negative_weeks:
        # datasets for prediction made at weeks with certain duration until peak
        for week_until_peak in (-4, -8, -12):

            build_dataset(week_of_prediction_in_fall=week_until_peak,
                          pred_period=(TIME_OF_FALL, SIM_DURATION),
                          hosp_threshold=HOSPITALIZATION_THRESHOLD)

            build_dataset(week_of_prediction_in_fall=week_until_peak,
                          pred_period=(TIME_OF_FALL, SIM_DURATION),
                          hosp_threshold=HOSPITALIZATION_THRESHOLD,
                          noise_coeff=1)

            build_dataset(week_of_prediction_in_fall=week_until_peak,
                          pred_period=(TIME_OF_FALL, SIM_DURATION),
                          hosp_threshold=HOSPITALIZATION_THRESHOLD,
                          noise_coeff=0.5, bias_delay=4)
