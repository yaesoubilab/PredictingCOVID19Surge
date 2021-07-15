import os
import pandas as pd


class DataEngineering:
    def __init__(self, directory_name, calib_period, proj_period, icu_capacity_rate):
        """ to create the dataset needed to develop the predictive models """
        self.directory_name = directory_name
        self.calib_period = calib_period
        self.proj_period = proj_period
        self.sim_duration = proj_period + calib_period
        self.hospitalization_threshold = icu_capacity_rate
        self.list_of_dataset_names = os.listdir(directory_name)

    def pre_processing(self, id_f_names_list, p_f_names_list, outcome_names_list):
        """
        read a trajectory in the assigned the directory and pre-process
        :param id_f_names_list: names of incidence feature names
        :param p_f_names_list: names of prevalence feature names
        :param outcome_names_list: names of outcomes to predict
        :return: pre-processed dataset
        """

        row_list = []
        for name in self.list_of_dataset_names:
            df = pd.read_csv('{}/{}'.format(self.directory_name, name))

            # create a new row based on this trajectory
            incidence_f = self._get_incidence_feature(df=df, feature_names=id_f_names_list)
            prevalence_f = self._get_prevalence_feature(df=df, feature_names=p_f_names_list)
            row = incidence_f + prevalence_f

            if_hosp_thred_passed, hosp_maximum = self._get_if_surpass_threshold_and_max(df=df)
            row.append(hosp_maximum)
            row.append(if_hosp_thred_passed)
            row_list.append(row)

        # convert to DataFrame
        df = pd.DataFrame(row_list, columns=(id_f_names_list + p_f_names_list + outcome_names_list))
        return df

    def _get_if_surpass_threshold_and_max(self, df):
        """
        :return: 'if threshold is passed' (0=no, 1=yes) and 'max hospitalization rate'
        """

        observation_time_list = df['Observation Time']
        hospitalization_rate_list = df['Obs: New hospitalization rate']

        # get maximum hospitalization rate during [calib_period, proj_period]
        maximum = 0
        for pair in zip(observation_time_list, hospitalization_rate_list):
            if self.calib_period <= pair[0] <= self.sim_duration:
                maximum = max(pair[1], maximum)

        # decide if surpass the hospitalization threshold
        if_surpass_threshold = 0
        if maximum > self.hospitalization_threshold:
            if_surpass_threshold = 1

        return if_surpass_threshold, maximum

    def _get_incidence_feature(self, df, feature_names):
        """
        get value of an incidence feature over the specified week
        :param df: df of interest
        :param feature_names: list of names of features that are observed over a week
        :return: list of values for incidence features
        """
        incidence_f_list = []
        for feature_name in feature_names:
            for pair in zip(df['Observation Period'], df[feature_name]):
                if pair[0] == 52 * self.calib_period:
                    incidence_f_list.append(pair[1])
        return incidence_f_list

    def _get_prevalence_feature(self, df, feature_names):
        """
        value of a prevalence feature at the specified time
        :param df: df of interest
        :param feature_names: features that are observed at the beginning of a week
        :return:
        """
        prevalence_f_list = []
        for feature_name in feature_names:
            for pair in zip(df['Observation Time'], df[feature_name]):
                if round(pair[0], 3) == self.calib_period:
                    prevalence_f_list.append(pair[1])
        return prevalence_f_list
