import os

import pandas as pd


class FeatureEngineering:
    def __init__(self, directory_name, calib_period, proj_period, hosp_threshold):
        """ to create the dataset needed to develop the predictive models """
        # TODO: I think instead of calib_period and proj_period we should get
        #   time_of_prediction and simulation_duration.
        #   Right now, the time when we'd like to make predictions is at the time when calibration period ends
        #   (i.e., time_of_prediction=calib_period) but later now we'd like to repeat these predictions
        #   maybe 4, 6, 8, ... weeks after the calib_period.
        self.directoryName = directory_name
        self.calibPeriod = calib_period
        self.simDuration = proj_period + calib_period
        self.hospThreshold = hosp_threshold
        self.datasetNames = os.listdir(directory_name)

    def pre_process(self, names_of_incidence_features, names_of_prevalence_features, file_name):
        """
        read a trajectory in the assigned the directory and pre-process
        :param names_of_incidence_features: names of incidence feature
        :param names_of_prevalence_features: names of prevalence feature
        """

        # TODO: we will need to use epidemic parameter values as features too. The list of parameter values
        #   for each trajectory is here: outputs\summary\parameter_values.csv.
        #   (row with ID 0 correspond to trajectory 0, row with ID 1 correspond to trajectory 1, etc.)
        #   You can add a new argument 'names_of_parameter_features' to read the values of
        #   selected parameters here (if you'd like to test it, read the values of 'R0s-0' parameter as a feature).

        row_list = []
        for name in self.datasetNames:
            df = pd.read_csv('{}/{}'.format(self.directoryName, name))

            # create a new row based on this trajectory
            # features
            incd_fs = self._get_incd_features(df=df, feature_names=names_of_incidence_features)
            prev_fs = self._get_prev_features(df=df, feature_names=names_of_prevalence_features)
            # outcomes to predict
            if_hosp_threshold_passed, hosp_max = self._get_if_threshold_passed_and_max(df=df)

            row = incd_fs + prev_fs
            row.extend([hosp_max, if_hosp_threshold_passed])
            row_list.append(row)

        # convert to DataFrame
        outcomes_labels = ['Maximum hospitalization rate', 'If hospitalization threshold passed']
        df = pd.DataFrame(data=row_list,
                          columns=(names_of_incidence_features + names_of_prevalence_features + outcomes_labels))

        # save new dataset to file
        df.to_csv(file_name, index=False)

    def _get_if_threshold_passed_and_max(self, df):
        """
        :return: 'if threshold is passed' (0=no, 1=yes) and 'max hospitalization rate'
        """

        observation_time_list = df['Observation Time']
        hospitalization_rate_list = df['Obs: New hospitalization rate']

        # get maximum hospitalization rate during [calib_period, proj_period]
        maximum = 0
        for pair in zip(observation_time_list, hospitalization_rate_list):
            if self.calibPeriod <= pair[0] <= self.simDuration:
                maximum = max(pair[1], maximum)

        # decide if surpass the hospitalization threshold
        if_surpass_threshold = 0
        if maximum > self.hospThreshold:
            if_surpass_threshold = 1

        return if_surpass_threshold, maximum

    def _get_incd_features(self, df, feature_names):
        """
        get value of an incidence feature over the specified week
        :param df: df of interest
        :param feature_names: list of names of features that are observed over a week
        :return: list of values for incidence features
        """
        incidence_f_list = []
        for feature_name in feature_names:
            for pair in zip(df['Observation Period'], df[feature_name]):
                if pair[0] == 52 * self.calibPeriod:
                    incidence_f_list.append(pair[1])
        return incidence_f_list

    def _get_prev_features(self, df, feature_names):
        """
        value of a prevalence feature at the specified time
        :param df: df of interest
        :param feature_names: features that are observed at the beginning of a week
        :return:
        """
        prevalence_f_list = []
        for feature_name in feature_names:
            for pair in zip(df['Observation Time'], df[feature_name]):
                if round(pair[0], 3) == self.calibPeriod:
                    prevalence_f_list.append(pair[1])
        return prevalence_f_list
