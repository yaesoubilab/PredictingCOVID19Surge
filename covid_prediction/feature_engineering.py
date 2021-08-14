import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUTCOME_LABELS = ['Maximum hospitalization rate', 'If hospitalization threshold passed']


class FeatureEngineering:
    def __init__(self, dir_of_trajs, week_of_prediction_in_fall, pred_period, hosp_threshold):
        """ create the dataset needed to develop the predictive models
        :param dir_of_trajs: (string) the name of directory where trajectories are located
        :param week_of_prediction_in_fall: (int) a positive int for number of weeks into fall and
                                                 a negative int for number of weeks before the peak
        :param pred_period: (tuple) (y0, y1) time (in year) when the prediction period starts and ends
        :param hosp_threshold: threshold of hospitalization capacity
        """
        self.directoryName = dir_of_trajs
        self.weekOfPredInFall = week_of_prediction_in_fall
        self.predictionPeriod = pred_period
        self.hospThreshold = hosp_threshold
        self.namesOfTrajFiles = os.listdir(dir_of_trajs)

    def pre_process(self, info_of_incd_fs, info_of_prev_fs, info_of_parameter_fs, output_file):
        """
        read a trajectory in the assigned the directory and pre-process
        :param info_of_incd_fs: information of incidence features
        :param info_of_prev_fs: information of prevalence features
        :param info_of_parameter_fs: names of parameter feature
        :param output_file: name of output csv
        """

        # find the labels of features
        # note that on a trajectory these features can be defined:
        # last recoding, average of last recordings, slope of last recordings.
        col_labels = []
        col_labels.extend(self._get_labels_of_incd_or_prev_features(info_of_incd_fs))
        col_labels.extend(self._get_labels_of_incd_or_prev_features(info_of_prev_fs))
        col_labels.extend(info_of_parameter_fs)
        col_labels.extend(OUTCOME_LABELS)

        # read dataset of the parameter features
        param_df = pd.read_csv('outputs/summary/parameter_values.csv')
        param_cols = []  # columns of parameter values
        for name in info_of_parameter_fs:
            param_cols.append(np.asarray(param_df[name]))

        # values of incidence, prevalence, and parameters features
        all_feature_values = []
        for i in range(len(self.namesOfTrajFiles)):

            # read trajectory file
            df = pd.read_csv('{}/{}'.format(self.directoryName, self.namesOfTrajFiles[i]))

            # find if for this trajectory threshold of hospitalization has passed, value of the peak, and
            # time of the peak
            if_hosp_threshold_passed, hosp_max, peak_week = \
                self._get_if_threshold_passed_and_max_and_week_of_peak(df=df)

            # find the time when feature values should be collected
            if self.weekOfPredInFall < 0:
                pred_week = peak_week - self.weekOfPredInFall
            else:
                pred_week = int(52*self.predictionPeriod[0]) + self.weekOfPredInFall

            # read values of incidence and prevalence features for this trajectory
            incd_fs = self._get_feature_values(df=df, info_of_features=info_of_incd_fs, incd_or_prev='incd')
            prev_fs = self._get_feature_values(df=df, info_of_features=info_of_prev_fs, incd_or_prev='prev')

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

        # report correlations
        report_corrs(df=df, outcomes=OUTCOME_LABELS,
                     csv_file_name='outputs/prediction_dataset/corrs-{}.csv'.format(output_file))

    def _get_if_threshold_passed_and_max_and_week_of_peak(self, df):
        """
        :return: 'if threshold is passed' (0=no, 1=yes) and 'max hospitalization rate', and 'week of the peak'
        """

        obs_times = df['Observation Time']
        obs_weeks = df['Observation Period']
        hosp_rates = df['Obs: New hospitalization rate']

        # get maximum hospitalization rate during the prediction period
        maximum = 0
        week = None
        for pair in zip(obs_times, obs_weeks, hosp_rates):
            if self.predictionPeriod[0] <= pair[0] <= self.predictionPeriod[1]:
                if pair[2] > maximum:
                    week = pair[1]
                    maximum = pair[2]

        # decide if surpass the hospitalization threshold
        if_surpass_threshold = 0
        if maximum > self.hospThreshold:
            if_surpass_threshold = 1

        return if_surpass_threshold, maximum, week

    def _get_feature_values(self, df, week, info_of_features, incd_or_prev):
        """
        get value of an incidence feature over the specified week
        :param df: df of interest
        :param week: (int) week when feature values should be collected
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
                        if pair[0] <= week:
                            data.append(pair[1])
                        else:
                            break
            elif incd_or_prev == 'prev':
                for pair in zip(df['Observation Time'], col):
                    if not np.isnan(pair[1]):
                        if 52 * pair[0] - week < 0.5:
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

        return f_values

    @staticmethod
    def _get_labels_of_incd_or_prev_features(info_of_incd_or_prev_fs):
        """
        :param info_of_incd_or_prev_fs: (list of strings or tuples)
            like: 'Obs: Prevalence susceptible' or
                  ('Obs: Incidence rate', ('ave', 2), ('slope', 4))
        """

        feature_names = []
        for info in info_of_incd_or_prev_fs:
            if isinstance(info, str):
                # feature name for the last recording
                feature_names.append(info)

            elif isinstance(info, tuple):
                for value in info:
                    if isinstance(value, str):
                        # feature for last recording
                        feature_names.append(info[0])
                    else:
                        # feature for average or slope
                        feature_names.append('{}-{}-{}wk'.format(info[0], value[0], value[1]))
            else:
                raise ValueError('Invalid feature type.')

        return feature_names


def report_corrs(df, outcomes, csv_file_name):

    col_labels = ['feature']
    for o in outcomes:
        col_labels.extend(['{} | corr'.format(o), '{} | p-value'.format(o)])

    # correlation between each feature column and outcomes
    rows = []
    for f_name in df:
        if f_name not in outcomes:
            row = [f_name]
            for o in outcomes:
                y = df[o]
                if f_name != o:
                    # correlation and p-value
                    corr, p = pearsonr(df[f_name], y)
                    row.extend([corr, p])
            rows.append(row)

    df = pd.DataFrame(data=rows,
                      columns=col_labels)
    df.to_csv(csv_file_name)
