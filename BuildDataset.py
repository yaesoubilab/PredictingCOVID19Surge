from covid_prediction.feature_engineering import *
from definitions import get_dataset_labels


HOSPITALIZATION_THRESHOLD = 0.0001  # per 100,000 population
TIME_OF_FALL = 1.5   # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
SIM_DURATION = 2.25

# survey sizes
N_NOVEL_INCD = 1521
N_PREV_SUSC = 481
BIAS_DELAY = 4


def build_dataset(week_of_prediction_in_fall, pred_period, hosp_threshold,
                  noise_coeff=None, bias_delay=None):
    """ create the dataset needed to develop the predictive models
    :param week_of_prediction_in_fall: (int) a positive int for number of weeks into fall and
                                             a negative int for number of weeks before the peak
    :param pred_period: (tuple) (y0, y1) time (in year) when the prediction period starts and ends
    :param hosp_threshold: threshold of hospitalization capacity
    :param noise_coeff: (None or int) if None, the noise model is not added, otherwise, the noise model is
        added with survey size multiplied by add_noise.
    :param bias_delay: (None or int): delay (in weeks) of observing the true value
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
    # if bias needs to be added
    if bias_delay:
        # and if noise needs to be added
        if noise_coeff is not None:
            err_novel_incd = ErrorModel(
                survey_size=N_NOVEL_INCD * noise_coeff, bias_delay=BIAS_DELAY)
            err_prev_susc = ErrorModel(
                survey_size=N_PREV_SUSC * noise_coeff, bias_delay=BIAS_DELAY)
    else: # no bias
        # if noise needs to be added
        if noise_coeff is not None:
            err_novel_incd = ErrorModel(survey_size=N_NOVEL_INCD * noise_coeff)
            err_prev_susc = ErrorModel(survey_size=N_PREV_SUSC * noise_coeff)

    # find output file name
    label = get_dataset_labels(
        week=week_of_prediction_in_fall, noise_coeff=noise_coeff, bias_delay=bias_delay)
    output_file = 'data-{}.csv'.format(label)

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


if __name__ == "__main__":
    for week_in_fall in (-4, -8, -12):

        build_dataset(week_of_prediction_in_fall=week_in_fall,
                      pred_period=(TIME_OF_FALL, SIM_DURATION),
                      hosp_threshold=HOSPITALIZATION_THRESHOLD)

        build_dataset(week_of_prediction_in_fall=week_in_fall,
                      pred_period=(TIME_OF_FALL, SIM_DURATION),
                      hosp_threshold=HOSPITALIZATION_THRESHOLD,
                      noise_coeff=0.5)

        build_dataset(week_of_prediction_in_fall=week_in_fall,
                      pred_period=(TIME_OF_FALL, SIM_DURATION),
                      hosp_threshold=HOSPITALIZATION_THRESHOLD,
                      noise_coeff=0.5, bias_delay=BIAS_DELAY)
