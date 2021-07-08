import pandas as pd
import os
from covid_prediction.DataDirectory import *


class DataEngineering:
    def __init__(self, directory_name, calib_period, proj_period, icu_capacity_rate):
        """ to create the dataset needed to develop the predictive models """
        self.directory_name = directory_name
        self.calib_period = calib_period
        self.proj_period = proj_period
        self.sim_duration = proj_period + calib_period
        self.icu_capacity_rate = icu_capacity_rate
        self.list_of_dataset_names = os.listdir(directory_name)
        self.num_datasets = len(self.list_of_dataset_names)

        # initiation
        self.output_dic = {}

    def read_datasets(self):
        """ read all csv in the assigned dictionary and save to a dictionary """
        i = 0
        for name in self.list_of_dataset_names:
            df = pd.read_csv('{}/{}'.format(self.directory_name, name))
            self.output_dic['df_{}'.format(i)] = df
            i += 1

    def pre_processing(self):
        """ create a combined dataset with observations from all trajectories """
        row_list = []
        # collect features from each of the trajectory dataset
        for i in range(self.num_datasets):
            # read a trajectory dataset
            df = self.output_dic['df_{}'.format(i)]
            # create a row based on this trajectory
            incidence = self._get_incidence_feature(df=df)
            vaccination = self._get_vaccine_feature(df=df)
            icu_capacity = self._whether_surpass_capacity(df=df)
            row = [incidence, vaccination, icu_capacity]
            row_list.append(row)
        # convert to DataFrame
        df = pd.DataFrame(row_list, columns=(FEATURES + [Y_NAME]))
        return df

    def _whether_surpass_capacity(self, df):
        """ return False if the ICU capacity rate never exceeds the threshold value, otherwise return True
            0 represents False, 1 represent True
        """
        surpass_capacity = 0
        observation_time_list = df['Observation Time']
        hospitalization_rate_list = df['Obs: Hospitalization rate']
        for pair in zip(observation_time_list, hospitalization_rate_list):
            # within simulation period, check hospitalization rate
            if self.calib_period <= pair[0] <= self.sim_duration:
                if pair[1] > self.icu_capacity_rate:
                    surpass_capacity = 1
            # within calibration period or after projection period, do nothing
        return surpass_capacity

    def _get_incidence_feature(self, df):
        """ incidence (at week D.CALIB_PERIOD * 52) """
        incidence = None
        for pair in zip(df['Observation Period'], df['Obs: Incidence']):
            if pair[0] == 52 * self.calib_period:
                incidence = pair[1]
        return incidence

    def _get_vaccine_feature(self, df):
        """ vaccination (at year D.CALIB_PERIOD) """
        vaccination = None
        for pair in zip(df['Observation Time'], df['Obs: Cumulative vaccination']):
            if round(pair[0], 3) == self.calib_period:
                vaccination = pair[1]
        return vaccination
