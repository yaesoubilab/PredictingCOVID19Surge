import os
from enum import Enum

DIGITS = 3
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# calibration settings
FEASIBILITY_PERIOD = 1.75  # years (from Mar-1, 2020 to Dec-1) -> week 91
CALIB_PERIOD = 2  # years (simulation duration during calibration)
N_SIM_CALIBRATION = 150*50  # number of simulated trajectories used for calibration

SIM_DURATION = 2.25  # years (until Jun-1, 2022)

# to build datasets for developing predictive models
FIRST_WEEK_OF_PREDICTION_PERIOD = 96  # for example, 96 is the first week of winter
WEEKS_TO_PREDICT = 8
HOSP_OCCU_THRESHOLDS = (10, 15, 20)  # per 100,000 population

# number of simulation runs used for training and validation
N_SIM_TRAINING = 20*50
N_SIM_VALIDATION = 5*50
CV_FOLD = 10         # num of splits for cross validation
FILL_TREE = True

SCENARIOS = {
    'base': 'base',
    'no control measure': 'no control measure',
    'no novel variant': 'no novel variant',
    'smaller survey': 'smaller survey'
}

# survey sizes
N_NOVEL_INCD = 1521
SMALLER_N_NOVEL_INCD = 250

# ------------------------------------------------------------------------------------
# ---- Other support variables, classes, and functions that should not be changed ----
# ------------------------------------------------------------------------------------

# columns in datasets
HOSP_OCCUPANCY_IN_TRAJ_FILE = 'Obs: Hospital occupancy rate'
OUTCOME_NAME_IN_DATASET = 'If threshold passed (0:Yes)'

AGES = ['0-4yrs', '5-12yrs', '13-17yrs', '18-29yrs', '30-49yrs', '50-64yrs', '65-74yrs', '75+yrs']
VARIANTS = ['Orig', 'Delta', 'Novel']
VACC_STATUS = ['UnVacc', 'Vacc']


class AgeGroups(Enum):
    Age_0_4 = 0
    Age_5_12 = 1
    Age_13_17 = 2
    Age_18_29 = 3
    Age_30_49 = 4
    Age_50_64 = 5
    Age_65_74 = 6
    Age_75_ = 7


class Variants(Enum):
    ORIGINAL = 0
    DELTA = 1
    NOVEL = 2


class ProfileDefiner:
    """ to convert (age group index, variant index, vaccination status index) to an index and vice versa """

    def __init__(self, n_age_groups, n_variants, n_vacc_status):
        self.nAgeGroups = n_age_groups
        self.nVariants = n_variants
        self.nVaccStatus = n_vacc_status
        self.nProfiles = self.nVariants * self.nVaccStatus
        self.length = n_age_groups * n_variants * n_vacc_status

        self.strAge = [None] * self.nAgeGroups
        self.strVariant = [None] * self.nVariants
        self.strProfile = [[None] * self.nVaccStatus for v in range(self.nVariants)]
        self.strAgeVariant = [[None] * self.nVariants for a in range(self.nAgeGroups)]
        self.strAgeProfile = [
            [[None] * self.nVaccStatus for v in range(self.nVariants)] for a in range(self.nAgeGroups)]

        for v in range(self.nVariants):
            self.strVariant[v] = VARIANTS[v]
            for vs in range(self.nVaccStatus):
                self.strProfile[v][vs] = VARIANTS[v] + '-' + VACC_STATUS[vs]

        for a in range(self.nAgeGroups):
            self.strAge[a] = AGES[a]
            for v in range(self.nVariants):
                self.strAgeVariant[a][v] = AGES[a] + '-' + VARIANTS[v]
                for vs in range(self.nVaccStatus):
                    self.strAgeProfile[a][v][vs] = AGES[a] + '-' + VARIANTS[v] + '-' + VACC_STATUS[vs]

    def get_row_index(self, age_group, variant, vacc_status):
        return self.nVariants * self.nVaccStatus * age_group + self.nVaccStatus * variant + vacc_status

    def get_profile_index(self, variant, vacc_status):
        return self.nVaccStatus * variant + vacc_status


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
