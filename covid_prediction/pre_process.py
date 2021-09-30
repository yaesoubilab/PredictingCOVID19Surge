import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from covid_prediction.feature_selection import rfe, lasso, pi


def standardize(x):
    return StandardScaler().fit_transform(x)


class PreProcessor:
    """ class to perform pre-processing steps """
    def __init__(self, df, feature_names, y_name):
        """
        :param df: (panda DataFrame)
        :param feature_names: (list) of feature names to be included
        :param y_name: (string) name of the outcome
        """

        self._df = df
        self._feature_names = feature_names
        self._yName = y_name
        self._X = np.asarray(self._df[self._feature_names])
        self._y = np.asarray(self._df[self._yName])

        # x, df, features after preprocessing
        # for now, we set them to the default values
        self.df = self._df
        self.X = self._X
        self.y = self._y.ravel()
        self.feature_name = self._feature_names

        # selected features and X after feature selection
        self.selectedFeatureNames = None
        self.selectedX = None

    def preprocess(self, y_is_binary=False, if_standardize=False, degree_of_polynomial=None):
        """
        :param y_is_binary: (bool) set True if outcome is a binary variable
        :param if_standardize: (bool) set True to standardize features and outcome
        :param degree_of_polynomial: (int >=1 ) to add polynomial terms
        """

        if if_standardize:
            self.X = standardize(self._X)
            if not y_is_binary:
                self.y = standardize(self._y.reshape(-1, 1)).ravel()

        if degree_of_polynomial is not None:
            poly = PolynomialFeatures(degree_of_polynomial)
            # polynomial is always done after standardization, so we work with X here
            self.X = poly.fit_transform(self.X)  # updating feature values
            self.feature_name = poly.get_feature_names(self._feature_names)  # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.feature_name)
        self.df[self._yName] = self.y

    def feature_selection(self, estimator, method, num_fs_wanted=10):
        """
        :param estimator: a supervised learning estimator with a fit method that provides
            information about feature importance
        :param method: 'rfe', 'lasso', and 'pi'
        :param num_fs_wanted: number of significant features want to select
        """

        if method == 'rfe':
            selected_features = rfe(x=self.X, y=self.y, features=self.feature_name,
                                    num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'pi':
            selected_features = pi(x=self.X, y=self.y, features=self.feature_name,
                                   num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'lasso':
            selected_features = lasso(x=self.X, y=self.y, features=self.feature_name,
                                      estimator=estimator)
        else:
            raise ValueError('unknown feature selection method')

        # update feature names
        self.selectedFeatureNames = selected_features
        # update predictor values
        self.selectedX = np.asarray(self.df[self.selectedFeatureNames])

