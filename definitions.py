import os
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIB_PERIOD = 1.5  # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
PROJ_PERIOD = 0.75  # year (from Sep-1, 2021 to May-31, 2022 which is 0.75 year)
SIM_DURATION = CALIB_PERIOD + PROJ_PERIOD

AGES = ['0-4yrs', '5-19yrs', '20-49yrs', '50-64yrs', '65-79yrs', '80+yrs']


class Profiles(Enum):
    A = 0   # infected with circulating strain
    B = 1   # infected with novel strain
    # V = 2   # vaccinated and infected with novel strain


class AgeGroups(Enum):
    Age_0_4 = 0
    Age_5_19 = 1
    Age_20_49 = 2
    Age_50_64 = 3
    Age_65_79 = 4
    Age_80_ = 5


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

        if profile == 0:
            return AGES[age_group] + '-Current'
        elif profile == 1:
            return AGES[age_group] + '-Novel'
        else:
            raise ValueError
        #return 'age/profile ({},{})'.format(age_group, profile)

    def get_str_age(self, age_group):
        return AGES[age_group]
        # return 'age {}'.format(age_group)
