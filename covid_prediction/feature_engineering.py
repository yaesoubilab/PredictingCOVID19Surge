import os
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy.stats import pearsonr

from SimPy.InOutFunctions import write_csv
from definitions import HOSP_OCCUPANCY_IN_TRAJ_FILE, OUTCOME_NAME_IN_DATASET, get_outcome_label


class ErrorModel:

    def __init__(self, survey_size=None, weeks_delay=0):
        """
        :param survey_size: (int) sample size 
        :param weeks_delay: (int) weeks of delay
        """

        self.surveySize = survey_size
        self.weeksDelay = weeks_delay
        self.rnd = RandomState(1)

    def get_obs(self, true_values):
        """
        :param true_values: (list) time-series of true values
        :return: observed value (with noise and bias added)
        """

        # which true value to use
        y = None
        if self.weeksDelay > 0:
            # the value with delay
            if len(true_values) > self.weeksDelay:
                y = true_values[-self.weeksDelay-1]
        else:
            # last true value
            y = true_values[-1]

        if y is not None:
            noise = self.get_noise(true_value=y, n=self.surveySize)
            return min(max(y + noise, 0), 1)
        else:
            return None

    def get_noise(self, true_value, n):

        if n is None:
            return 0
        else:
            st_dev = sqrt(true_value * (1 - true_value) / n)
            return self.rnd.normal(loc=0, scale=st_dev)


class FeatureEngineering:
    def __init__(self, dir_of_trajs, weeks_of_pred_period, weeks_to_predict,
                 hosp_thresholds, n_of_trajs_used=None):
        """ create the dataset needed to develop the predictive models
        :param dir_of_trajs: (string) the name of directory where trajectories are located
        :param weeks_of_pred_period: (tuple) (y0, y1) weeks when the prediction period starts and ends
        :weeks_to_predict: (int) number of weeks to predict in the future
        :param hosp_thresholds: (list) of thresholds for hospitalization capacity
        :param n_of_trajs_used: (None or int) number of trajectories used to build the dataset
            (if None, all trajectories are used)
        """
        self.directoryName = dir_of_trajs
        self.weeksOfPredictionPeriod = weeks_of_pred_period
        self.weeksToPredict = weeks_to_predict
        self.hospThresholds = hosp_thresholds
        if n_of_trajs_used is None:
            self.namesOfTrajFiles = os.listdir(dir_of_trajs)
        else:
            self.namesOfTrajFiles = os.listdir(dir_of_trajs)[:n_of_trajs_used]

    def pre_process(self, info_of_incd_fs, info_of_prev_fs, info_of_parameter_fs, output_file, report_corr=True):
        """
        read a trajectory in the assigned the directory and pre-process
        :param info_of_incd_fs: information of incidence features
        :param info_of_prev_fs: information of prevalence features
        :param info_of_parameter_fs: names of parameter feature
        :param output_file: name of output csv
        :param report_corr: (bool) whether to report correlations between features and outcomes
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
                  file_name='outputs/prediction_datasets_{}_weeks/features.csv'.format(self.weeksToPredict))

        # add columns for outcomes
        col_labels.append('Max ' + HOSP_OCCUPANCY_IN_TRAJ_FILE)
        for t in self.hospThresholds:
            col_labels.append(get_outcome_label(threshold=t))

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

            # read values of incidence and prevalence features for this trajectory
            incd_fs = self._get_feature_values(
                df=df, week=self.weeksOfPredictionPeriod[0],
                info_of_features=info_of_incd_fs, incd_or_prev='incd')
            prev_fs = self._get_feature_values(
                df=df, week=self.weeksOfPredictionPeriod[0],
                info_of_features=info_of_prev_fs, incd_or_prev='prev')

            # make a row of feature values
            # incidence features, prevalence features
            row = incd_fs + prev_fs
            # add epidemic parameter values for corresponding trajectory
            for col in param_cols:
                row.append(col[i])
            # max hospital rate and whether surpass capacity
            row.append(hosp_max)
            row.extend(if_hosp_threshold_passed)

            # store this row of feature values
            all_feature_values.append(row)

        # convert to DataFrame
        df = pd.DataFrame(data=all_feature_values,
                          columns=col_labels)

        # find directoy
        output_dir = Path('outputs/prediction_datasets_{}_weeks/'.format(self.weeksToPredict))

        output_dir.mkdir(parents=True, exist_ok=True)

        # save new dataset to file
        df.to_csv(output_dir / output_file, index=False)

        # report correlations
        if report_corr:
            report_corrs(df=df, outcomes=OUTCOME_NAME_IN_DATASET,
                         csv_file_name=output_dir / 'corrs-{}'.format(output_file))

    def _get_if_threshold_passed_and_max_and_week_of_peak(self, df):
        """
        :return: 'if threshold is passed' (0=no, 1=yes) and 'max hospitalization rate', and 'week of the peak'
        """

        obs_times = df['Observation Time']
        obs_weeks = df['Observation Period']
        # hosp_rates = df['Obs: New hospitalization rate']
        hosp_occu_rates = df[HOSP_OCCUPANCY_IN_TRAJ_FILE]

        # get maximum hospitalization rate during the prediction period
        maximum = 0
        week_of_peak = None
        for pair in zip(obs_times, obs_weeks, hosp_occu_rates):
            if self.weeksOfPredictionPeriod[0] <= pair[1] < self.weeksOfPredictionPeriod[1]:
                if pair[2] > maximum:
                    week_of_peak = pair[1]
                    maximum = pair[2]
            # exit loop if prediction period has passed
            if pair[1] > self.weeksOfPredictionPeriod[1]:
                break

        # decide if surpass the hospitalization threshold
        # 0 if yes, 1 if not
        if_surpass_thresholds = [1] * len(self.hospThresholds)
        for i, t in enumerate(self.hospThresholds):
            if maximum * 100000 > t:
                if_surpass_thresholds[i] = 0

        return if_surpass_thresholds, maximum, week_of_peak

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

        err_model = None  # error model
        f_values = []   # feature values
        for info in info_of_features:
            # multiplier to multiply the value of this column by
            multiplier = 1

            # get the column in trajectory files where the data is located to define features
            if isinstance(info, str):
                col = df[info]
            elif isinstance(info, tuple):
                # find multiplier
                if type(info[1]) == int or type(info[1]) == float:
                    multiplier = info[1]

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
                        if pair[0] < week:
                            true_values.append(pair[1]*multiplier)
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
                            true_values.append(pair[1]*multiplier)
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
                            if None in y:
                                slope = 0
                            else:
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
                    row.extend([round(corr, 3), round(p, 3)])
            rows.append(row)

    df = pd.DataFrame(data=rows,
                      columns=col_labels)
    df.to_csv(csv_file_name, index=False)

