import os
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CALIB_PERIOD = 1.5  # year (from Mar-1, 2020 to Aug-31, 2021 which is 1.5 years)
PROJ_PERIOD = 0.75  # year (from Sep-1, 2021 to May-31, 2022 which is 0.75 year)
SIM_DURATION = CALIB_PERIOD + PROJ_PERIOD

AGES = ['0-4yrs', '5-12yrs', '13-17yrs', '18-29yrs', '30-49yrs', '50-64yrs', '65-74yrs', '75+yrs']
PROFILES = ['Dominant', 'Novel', 'Vaccinated']


class Profiles(Enum):
    D = 0   # infected with dominant strain
    N = 1   # infected with novel strain
    V = 2   # vaccinated and infected with novel strain


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
