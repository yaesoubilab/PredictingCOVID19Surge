import os
from enum import Enum

DIGITS = 3
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

FEASIBILITY_PERIOD = 1.5 # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
CALIB_PERIOD = 2  #
PROJ_PERIOD = 0.25  # year (from Sep-1, 2021 to May-31, 2022 which is 0.75 year)
SIM_DURATION = CALIB_PERIOD + PROJ_PERIOD

# to build datasets for developing predictive models
WEEKS_TO_PREDICT = 6
WEEKS_IN_FALL = (8, 12, 16, 20, 24, 28, 32)
HOSP_OCCU_THRESHOLDS = (5, 7.5, 10, 15, 20)  # per 100,000 population

# number of simulation runs used for calibration, training and validation
N_SIM_CALIBRATION = 1200*10
N_SIM_TRAINING = 40*10
N_SIM_VALIDATION = 10*10
CV_FOLD = 20         # num of splits for cross validation


SCENARIOS = {
    'base': 'base',
    'no control measure': 'no control measure',
    'no novel variant': 'no novel variant',
    'smaller survey': 'smaller survey'
}

# survey sizes
N_NOVEL_INCD = 1521
SMALLER_N_NOVEL_INCD = 250

# columns in datasets
HOSP_OCCUPANCY_IN_TRAJ_FILE = 'Obs: Hospital occupancy rate'
OUTCOME_NAME_IN_DATASET = 'If threshold passed (0:Yes)'

AGES = ['0-4yrs', '5-12yrs', '13-17yrs', '18-29yrs', '30-49yrs', '50-64yrs', '65-74yrs', '75+yrs']
PROFILES = ['Dominant-UV', 'Novel-UV',
            'Dominant-V', 'Novel-V']


class Profiles(Enum):
    DOM_UNVAC = 0   # infected with dominant strain
    NOV_UNVAC = 1   # infected with novel strain
    DOM_VAC = 2  # vaccinated and infected with dominant strain
    NOV_VAC = 3   # vaccinated and infected with novel strain


class AgeGroups(Enum):
    Age_0_4 = 0
    Age_5_12 = 1
    Age_13_17 = 2
    Age_18_29 = 3
    Age_30_49 = 4
    Age_50_64 = 5
    Age_65_74 = 6
    Age_75_ = 7


class AgeGroupsProfiles:
    # to convert (age group index, profile index) to an index and vice versa

    def __init__(self, n_age_groups, n_profiles):
        self.nAgeGroups = n_age_groups
        self.nProfiles = n_profiles
        self.length = n_age_groups * n_profiles

    def get_row_index(self, age_group, profile):
        return self.nProfiles * age_group + profile

    def get_age_group_and_profile(self, i):
        return int(i/self.nProfiles), i % self.nAgeGroups

    def get_str_age_profile(self, age_group, profile):
        return AGES[age_group] + '-' + PROFILES[profile]

    def get_str_age(self, age_group):
        return AGES[age_group]

    def get_str_profile(self, profile):
        return PROFILES[profile]


def get_dataset_labels(week, survey_size=None, bias_delay=None):

    if bias_delay is not None and survey_size is not None:
        label = ' sample size {} and bias {}'.format(survey_size, bias_delay)
    elif survey_size is not None:
        label = ' sample size {}'.format(survey_size)
    else:
        label = ''

    if week is not None:
        label = 'wk {}'.format(week) + label

    return label


def get_short_outcome(outcome):

    if outcome == 'Maximum hospitalization rate':
        return 'size'
    elif outcome == 'If hospitalization threshold passed':
        return 'prob'
    else:
        raise ValueError('Invalid outcome to predict.')


def get_outcome_label(threshold):

    return OUTCOME_NAME_IN_DATASET + ' {}'.format(threshold)
