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
        self.names_of_traj_files = os.listdir(directory_name)

    def pre_process(self, info_of_incd_fs, info_of_prev_fs, info_of_parameter_fs, output_file):
        """
        read a trajectory in the assigned the directory and pre-process
        :param info_of_incd_fs: information of incidence features
        :param info_of_prev_fs: information of prevalence features
        :param info_of_parameter_fs: names of parameter feature
        :param output_file: name of output csv
        """

        outcomes_labels = ['Maximum hospitalization rate', 'If hospitalization threshold passed']

        # find the labels of features
        # (note that on a trajectory these features can be defined:
        # last recoding, average of last recordings, slope of last recordings.
        col_labels = []
        col_labels.extend(self._get_labels_of_incd_or_prev_features(info_of_incd_fs))
        col_labels.extend(self._get_labels_of_incd_or_prev_features(info_of_prev_fs))
        col_labels.extend(info_of_parameter_fs)
        col_labels.extend(outcomes_labels)

        # read dataset of the parameter features
        param_df = pd.read_csv('outputs/summary/parameter_values.csv')
        param_cols = []  # columns of parameter values
        for name in info_of_parameter_fs:
            param_cols.append(np.asarray(param_df[name]))

        # values of incidence, prevalence, and parameters features
        all_feature_values = []
        for i in range(len(self.names_of_traj_files)):

            # read trajectory file
            df = pd.read_csv('{}/{}'.format(self.directoryName, self.names_of_traj_files[i]))

            # read values of incidence and prevalence features for this trajectory
            incd_fs = self._get_feature_values(df=df, info_of_features=info_of_incd_fs, incd_or_prev='incd')
            prev_fs = self._get_feature_values(df=df, info_of_features=info_of_prev_fs, incd_or_prev='prev')

            # values of outcomes to predict
            if_hosp_threshold_passed, hosp_max = self._get_if_threshold_passed_and_max(df=df)

            # make a row of feature values
            # incidence features, prevalence features
            row = incd_fs + prev_fs
            # add epidemic parameter values for corresponding trajectory
            for col in param_cols:
                row.append(col[i])
            # max hospital rate and whether surpass capacity
            row.extend([hosp_max, if_hosp_threshold_passed])

            # store this row of feature values
            all_feature_values.append(row)

        # convert to DataFrame
        df = pd.DataFrame(data=all_feature_values,
                          columns=col_labels)

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

    def _get_feature_values(self, df, info_of_features, incd_or_prev):
        """
        get value of an incidence feature over the specified week
        :param df: df of interest
        :param info_of_features: list of information for features that are observed over a week
        :param incd_or_prev: 'incd' or 'prev' to specify if incidence features or prevalence features are provided
        :return: list of values for features
        """

        f_values = []   # feature values
        for info in info_of_features:
            # get the column in trajectory files where the data is located to define features
            if isinstance(info, str):
                col = df[info]
            elif isinstance(info, tuple):
                col = df[info[0]]
            else:
                raise ValueError('Invalid feature information.')

            # read trajectory data until the time of prediction
            data = []
            if incd_or_prev == 'incd':
                for pair in zip(df['Observation Period'], col):
                    if not np.isnan(pair[1]):
                        if pair[0] <= 52 * self.timeOfPrediction:
                            data.append(pair[1])
                        else:
                            break
            elif incd_or_prev == 'prev':
                for pair in zip(df['Observation Time'], col):
                    if not np.isnan(pair[1]):
                        if 52 * (pair[0] - self.timeOfPrediction) < 0.5:
                            data.append(pair[1])
                        else:
                            break
            else:
                raise ValueError('Invalid value for the type of features.')

            # calculate feature value
            if isinstance(info, str):
                f_values.append(data[-1])
            elif isinstance(info, tuple):
                for v in info:
                    if isinstance(v, str):
                        f_values.append(data[-1])
                    elif isinstance(v, tuple):
                        if v[0] == 'ave':
                            f_values.append(np.average(data[-v[1]:]))
                        elif v[0] == 'slope':
                            x = np.arange(0, v[1])
                            y = data[-v[1]:]
                            slope = np.polyfit(x, y, deg=1)[0]
                            f_values.append(slope)
                        else:
                            raise ValueError('Invalid.')
            else:
                raise ValueError('Invalid feature information.')

                # # feature name for last recording
                # for pair in zip(df['Observation Period'], df[info]):
                #     if pair[0] == 52 * self.timeOfPrediction:
                #         incidence_f_list.append(pair[1])
                #         break

        return f_values

    @staticmethod
    def _get_labels_of_incd_or_prev_features(info_of_incd_or_prev_fs):

        feature_names = []
        for info in info_of_incd_or_prev_fs:
            if isinstance(info, str):
                # feature name for last recording
                feature_names.append(info)

            elif isinstance(info, tuple):
                # feature for last recording

                for value in info:
                    if isinstance(value, str):
                        feature_names.append(info[0])
                    else:
                        feature_names.append('{}-{}-{}wk'.format(info[0], value[0], value[1]))
            else:
                raise ValueError('Invalid feature type.')

        return feature_names
