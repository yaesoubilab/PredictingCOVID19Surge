import os
from pathlib import Path

import numpy as np
import pandas as pd


class FeatureEngineering:
    def __init__(self, directory_name, time_of_prediction, sim_duration, hosp_threshold):
        """ create the dataset needed to develop the predictive models
        :param directory_name: name of the trajectory directory
        :param time_of_prediction: (year) starting time of dataset used for prediction
        :param sim_duration: (year) simulation duration
        :param hosp_threshold: threshold of hospital capacity
        """
        self.directoryName = directory_name
        self.timeOfPrediction = time_of_prediction
        self.simDuration = sim_duration
        self.hospThreshold = hosp_threshold
        self.datasetNames = os.listdir(directory_name)

    def pre_process(self, names_of_incd_fs, names_of_prev_fs, names_of_parameter_fs, output_file):
        """
        read a trajectory in the assigned the directory and pre-process
        :param names_of_incd_fs: names of incidence feature
        :param names_of_prev_fs: names of prevalence feature
        :param names_of_parameter_fs: names of epidemic feature
        :param output_file: name of output csv
        """

        # read dataset of epidemic features
        param_df = pd.read_csv('outputs/summary/parameter_values.csv')
        param_col = []
        for name in names_of_parameter_fs:
            param_col.append(np.asarray(param_df[name]))

        dataset = []
        for i in range(len(self.datasetNames)):
            df = pd.read_csv('{}/{}'.format(self.directoryName, self.datasetNames[i]))

            # create a new row based on this trajectory
            # features
            incd_fs = self._get_incd_features(df=df, feature_names=names_of_incd_fs)
            prev_fs = self._get_prev_features(df=df, feature_names=names_of_prev_fs)
            # outcomes to predict
            if_hosp_threshold_passed, hosp_max = self._get_if_threshold_passed_and_max(df=df)

            # incidence features, prevalence features
            traj_row = incd_fs + prev_fs
            # add epidemic parameter values for corresponding trajectory
            for col in param_col:
                traj_row.append(col[i])
            # max hospital rate and whether surpass capacity
            traj_row.extend([hosp_max, if_hosp_threshold_passed])

            dataset.append(traj_row)

        # convert to DataFrame
        outcomes_labels = ['Maximum hospitalization rate', 'If hospitalization threshold passed']
        df = pd.DataFrame(data=dataset,
                          columns=(names_of_incd_fs + names_of_prev_fs + names_of_parameter_fs + outcomes_labels))

        # save new dataset to file
        output_dir = Path('outputs/prediction_dataset/')
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / output_file, index=False)

    def _get_if_threshold_passed_and_max(self, df):
        """
        :return: 'if threshold is passed' (0=no, 1=yes) and 'max hospitalization rate'
        """

        observation_time_list = df['Observation Time']
        hospitalization_rate_list = df['Obs: New hospitalization rate']

        # get maximum hospitalization rate during [calib_period, proj_period]
        maximum = 0
        for pair in zip(observation_time_list, hospitalization_rate_list):
            if self.timeOfPrediction <= pair[0] <= self.simDuration:
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
                if pair[0] == 52 * self.timeOfPrediction:
                    incidence_f_list.append(pair[1])
                    break
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
                if 52*abs(pair[0] - self.timeOfPrediction) < 0.5:
                    prevalence_f_list.append(pair[1])
                    break
        return prevalence_f_list
