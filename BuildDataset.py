from covid_prediction.feature_engineering import *


HOSPITALIZATION_THRESHOLD = 0.0001  # per 100,000 population
TIME_OF_FALL = 1.5   # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
SIM_DURATION = 2.25

# survey sizes
N_NOVEL_INCD = 690
N_PREV_SUSC = 500


def build_dataset(week_of_prediction_in_fall, pred_period, hosp_threshold, add_noise=False, add_noise_bias=False):
    """ create the dataset needed to develop the predictive models
    :param week_of_prediction_in_fall: (int) a positive int for number of weeks into fall and
                                             a negative int for number of weeks before the peak
    :param pred_period: (tuple) (y0, y1) time (in year) when the prediction period starts and ends
    :param hosp_threshold: threshold of hospitalization capacity
    :param add_noise: (bool) if noise models should be added
    :param add_noise_bias: (bool) if noise and bias models should be added
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
    if add_noise:
        err_novel_incd = ErrorModel(survey_size=N_NOVEL_INCD)
        err_prev_susc = ErrorModel(survey_size=N_PREV_SUSC)

    # find output file name
    if add_noise_bias:
        output_file = 'data at wk {} with noise and bias.csv'.format(week_of_prediction_in_fall)
    elif add_noise:
        output_file = 'data at wk {} with noise.csv'.format(week_of_prediction_in_fall)
    else:
        output_file = 'data at wk {}.csv'.format(week_of_prediction_in_fall)

    # create new dataset based on raw data
    feature_engineer.pre_process(
        # information for incidence and prevalence features can be provided as
        #   'name of the feature' to calculate the recording at prediction time
        # or
        #   (feature's name, ('ave', 2), ('slope', 4))
        # to report the recording at prediction time and
        # to calculate the average and slope of observations during past weeks
        info_of_incd_fs=[
            ('Obs: Incidence rate', ('ave', 2), ('slope', 4)),
            ('Obs: New hospitalization rate', ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence due to Novel-Unvaccinated', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence due to Novel-Vaccinated', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to Novel-Unvaccinated', ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations due to Novel-Vaccinated', ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence with novel variant', err_novel_incd, ('ave', 2), ('slope', 4)),
            ('Obs: % of new hospitalizations with novel variant', ('ave', 2), ('slope', 4)),
        ],
        info_of_prev_fs=[
            'Obs: Cumulative hospitalization rate',
            'Obs: Cumulative vaccination rate',
            ('Obs: Prevalence susceptible', err_prev_susc)],
        info_of_parameter_fs=[
            'R0',
            'Ratio transmissibility by profile-1',
            'Ratio transmissibility by profile-2',
            'Duration of infectiousness-dominant',
            'Ratio of infectiousness duration by profile-1',
            'Ratio of infectiousness duration by profile-2',
            'Prob Hosp for 18-29',
            'Relative prob hosp by age-0',
            'Relative prob hosp by age-1',
            'Relative prob hosp by age-2',
            'Relative prob hosp by age-4',
            'Relative prob hosp by age-5',
            'Relative prob hosp by age-6',
            'Relative prob hosp by age-7',
            'Ratio of hospitalization probability by profile-1',
            'Ratio of hospitalization probability by profile-2',
            'Duration of R-0',
            'Duration of R-1',
            'Duration of R-2',
            'Vaccine effectiveness against infection with novel',
            'Duration of vaccine immunity',
            'Prob novel strain params-0',
            'Prob novel strain params-1',
            'Prob novel strain params-2',
            'PD Y1 thresholds-0',
            'PD Y1 thresholds-1',
            'Change in contacts - PD Y1'
        ],
        output_file=output_file)


# create datasets for different prediction times
# for week_in_fall in (8, 12, 16):
#     build_dataset(week_of_prediction_in_fall=week_in_fall,
#                   pred_period=(TIME_OF_FALL, SIM_DURATION),
#                   hosp_threshold=HOSPITALIZATION_THRESHOLD)

for week_in_fall in (-4, -8, -12):
    # build_dataset(week_of_prediction_in_fall=week_in_fall,
    #               pred_period=(TIME_OF_FALL, SIM_DURATION),
    #               hosp_threshold=HOSPITALIZATION_THRESHOLD)

    build_dataset(week_of_prediction_in_fall=week_in_fall,
                  pred_period=(TIME_OF_FALL, SIM_DURATION),
                  hosp_threshold=HOSPITALIZATION_THRESHOLD,
                  add_noise=True)
