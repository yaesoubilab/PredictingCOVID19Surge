import os

import pandas as pd

from covid_prediction.DataDirectory import *


class DataEngineering:
    def __init__(self, directory_name, calib_period, proj_period, icu_capacity_rate):
        """ to create the dataset needed to develop the predictive models """
        self.directory_name = directory_name
        self.calib_period = calib_period
        self.proj_period = proj_period
        self.sim_duration = proj_period + calib_period
        self.hospitalization_threshold = icu_capacity_rate
        self.list_of_dataset_names = os.listdir(directory_name)
        self.num_datasets = len(self.list_of_dataset_names)

        # initiation
        self.output_dic = {}

    def read_datasets(self):
        """ read all csv in the assigned dictionary and save to a dictionary """
        # TODO: for the final analysis we probably need thousands of trajectories so
        #   it may make more sense to do the pre-processing at the time when we read a trajectory
        #   (that way we can avoid keeping the entire data-frame of the trajectory in the memory).

        i = 0
        for name in self.list_of_dataset_names:
            df = pd.read_csv('{}/{}'.format(self.directory_name, name))
            self.output_dic['df_{}'.format(i)] = df
            i += 1

    def pre_processing(self):
        """ create a combined dataset with observations from all trajectories """

        # TODO: to explore other features, can we modify this function to take 3 arguments:
        #   list_of_incidence_features: names of incidence features,
        #   list_of_prevalence_features: names of prevalence features,
        #   list_of_outcomes: names of outcomes to predict.
        #   See my comments below for the difference between incidence and prevalence features
        #

        row_list = []
        # collect features from each of the trajectory dataset
        for i in range(self.num_datasets):
            # read a trajectory dataset
            df = self.output_dic['df_{}'.format(i)]

            # create a row based on this trajectory
            incidence = self._get_incidence_feature(
                df=df, feature_name='Obs: Incidence', week=52 * self.calib_period)
            vaccination = self._get_prevalence_feature(
                df=df, feature_name='Obs: Cumulative vaccination', time=self.calib_period)

            # TODO: in addition to "whether we expect to pass the hospitalization threshold", we are also
            #   interested in the maximum hospitalization rate.
            #   Would you please modify this function to return both
            #   'if threshold passed' and 'max hospitalization rate'
            #   to be used as outcomes in the predictive models?
            if_hosp_threshold_passed, maximum = self._get_if_surpass_threshold_and_max(df=df)

            row = [incidence, vaccination, if_hosp_threshold_passed]
            row_list.append(row)

        # convert to DataFrame
        df = pd.DataFrame(row_list, columns=(FEATURES + [Y_NAME]))
        return df

    def _get_if_surpass_threshold_and_max(self, df):
        """ return 0 if the hospitalization rate never exceeds the threshold value, otherwise return 1 """

        # TODO: to get both 'if threshold passed' and 'max hospitalization rate' I think an efficient
        #   way to first calculate the maximum and then compare the maximum with the threshold
        #   to decide whether the threshold has passed.

        maximum = float('int')  # TODO: to be calculated below
        if_surpass_threshold = 0
        observation_time_list = df['Observation Time']
        hospitalization_rate_list = df['Obs: Hospitalization rate']

        for pair in zip(observation_time_list, hospitalization_rate_list):
            # within simulation period, check hospitalization rate
            if self.calib_period <= pair[0] <= self.sim_duration:
                if pair[1] > self.hospitalization_threshold:
                    if_surpass_threshold = 1
            # within calibration period or after projection period, do nothing

        return if_surpass_threshold, maximum

    # TODO: I tried to make some changes so that the following two functions
    #  can work with other features as well. Check to make sure I didn't mess anything up.
    #  Also, note that we can divide features into two groups:
    #   features that are observed over a week (incidence feature), and
    #   features that are observed at the beginning of a week (prevalence features).
    def _get_incidence_feature(self, df, feature_name, week):
        """ value of an incidence feature over the specified week """

        incidence = None
        for pair in zip(df['Observation Period'], df[feature_name]):
            if pair[0] == week:
                incidence = pair[1]
        return incidence

    def _get_prevalence_feature(self, df, feature_name, time):
        """ value of a prevalence feature at the specified time """
        vaccination = None
        for pair in zip(df['Observation Time'], df[feature_name]):
            if round(pair[0], 3) == time:
                vaccination = pair[1]
        return vaccination
