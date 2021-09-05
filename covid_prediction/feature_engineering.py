import os
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy.stats import pearsonr

from SimPy.InOutFunctions import write_csv

OUTCOME_LABELS = ['Maximum hospitalization rate', 'If hospitalization threshold passed']


class ErrorModel:

    def __init__(self, survey_size=None, bias_delay=None):

        self.surveySize = survey_size
        self.biasDelay = bias_delay
        self.rnd = RandomState(1)

    def get_obs(self, true_values):
        """
        :param true_values: (list) time-series of true values
        :return: observed value (with noise and bias added)
        """

        bias = 0
        noise = 0

        # bias
        if self.biasDelay is not None:
            # if enough observations are accumulated
            if len(true_values) >= self.biasDelay:
                bias = true_values[-self.biasDelay] - true_values[-1]
                noise = self.get_noise(true_value=true_values[-self.biasDelay], n=self.surveySize)
            else:
                bias = 0

        else:
            noise = self.get_noise(true_value=true_values[-1], n=self.surveySize)

        return min(max(true_values[-1] + bias + noise, 0), 1)

    def get_noise(self, true_value, n):

        if n is None:
            return 0
        else:
            st_dev = sqrt(true_value * (1 - true_value) / n)
            return self.rnd.normal(loc=0, scale=st_dev)


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
        self.predictionPeriodWeek = (int(pred_period[0]*52), int(pred_period[1]*52))
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
        # print feature names
        write_csv(rows=[[c] for c in col_labels],
                  file_name='outputs/prediction_datasets/features.csv')

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
                pred_week = peak_week + self.weekOfPredInFall
            else:
                pred_week = self.predictionPeriodWeek[0] + self.weekOfPredInFall

            # read values of incidence and prevalence features for this trajectory
            incd_fs = self._get_feature_values(df=df, week=pred_week,
                                               info_of_features=info_of_incd_fs, incd_or_prev='incd')
            prev_fs = self._get_feature_values(df=df, week=pred_week,
                                               info_of_features=info_of_prev_fs, incd_or_prev='prev')

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
        output_dir = Path('outputs/prediction_datasets/')
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / output_file, index=False)

        # report correlations
        report_corrs(df=df, outcomes=OUTCOME_LABELS,
                     csv_file_name='outputs/prediction_datasets/corrs-{}'.format(output_file))

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
            if self.predictionPeriodWeek[0] <= pair[1] <= self.predictionPeriodWeek[1]:
                if pair[2] > maximum:
                    week = pair[1]
                    maximum = pair[2]

        # decide if surpass the hospitalization threshold
        if_surpass_threshold = 0
        if maximum > self.hospThreshold:
            if_surpass_threshold = 1

        return if_surpass_threshold, maximum, week

    @staticmethod
    def _get_feature_values(df, week, info_of_features, incd_or_prev):
        """
        get value of an incidence feature over the specified week
        :param df: df of interest
        :param week: (int) week when feature values should be collected
        :param info_of_features: list of information for features that are observed over a week
        :param incd_or_prev: 'incd' or 'prev' to specify if incidence features or prevalence features are provided
        :return: list of values for features
        """

        err_model = None # error model
        f_values = []   # feature values
        for info in info_of_features:
            # get the column in trajectory files where the data is located to define features
            if isinstance(info, str):
                col = df[info]
            elif isinstance(info, tuple):
                # feature name
                col = df[info[0]]
                # find the error model
                for v in info:
                    if isinstance(v, ErrorModel):
                        err_model = v
            else:
                raise ValueError('Invalid feature information.')

            # read trajectory data until the time of prediction
            true_values = []
            observed_values = []
            if incd_or_prev == 'incd':
                for pair in zip(df['Observation Period'], col):
                    if not np.isnan(pair[1]):
                        if pair[0] <= week:
                            true_values.append(pair[1])
                            if err_model is None:
                                observed_values.append(true_values[-1])
                            else:
                                observed_values.append(err_model.get_obs(true_values=true_values))
                        else:
                            break
            elif incd_or_prev == 'prev':
                for pair in zip(df['Observation Time'], col):
                    if not np.isnan(pair[1]):
                        if 52 * pair[0] - week < 0.5:
                            true_values.append(pair[1])
                            if err_model is None:
                                observed_values.append(true_values[-1])
                            else:
                                observed_values.append(err_model.get_obs(true_values=true_values))
                        else:
                            break
            else:
                raise ValueError('Invalid value for the type of features.')

            # calculate feature value
            if isinstance(info, str):
                f_values.append(observed_values[-1])
            elif isinstance(info, tuple):
                for v in info:
                    if isinstance(v, str):
                        # get the last observation
                        f_values.append(observed_values[-1])
                    elif isinstance(v, tuple):
                        if v[0] == 'ave':
                            # get the average
                            f_values.append(np.average(observed_values[-v[1]:]))
                        elif v[0] == 'slope':
                            # get the slope
                            x = np.arange(0, v[1])
                            y = observed_values[-v[1]:]
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
                    elif isinstance(value, tuple):
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
