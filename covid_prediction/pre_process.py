import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from covid_prediction.feature_selection import rfe, lasso, pi


def standardize(x):
    return StandardScaler().fit_transform(x)


class PreProcessor:
    def __init__(self, df, features, y_name):

        self._df = df
        self._features = features
        self._yName = y_name
        self._X = np.asarray(self._df[self._features])
        self._y = np.asarray(self._df[self._yName])

        # x, df, features after preprocess
        self.df = self._df
        self.X = self._X        # X after standardization
        self.y = self._y.ravel()
        self.features = self._features

        # selected features and X after feature selection
        self.selectedFeatures = None
        self.selectedX = None

    def preprocess(self, if_standardize=False, degree_of_polynomial=None):
        """
        :param if_standardize: (bool) set True to standardize features and outcome
        :param degree_of_polynomial: (int >=1 ) to add polynomial terms
        """

        if if_standardize:
            self.X = standardize(self._X)
            self.y = standardize(self._y.reshape(-1, 1)).ravel()

        if degree_of_polynomial is not None:
            poly = PolynomialFeatures(degree_of_polynomial)
            # polynomial is always done after standardization, so we work with X here
            self.X = poly.fit_transform(self.X)  # updating feature values
            self.features = poly.get_feature_names(self._features)  # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.features)
        self.df[self._yName] = self.y

    def feature_selection(self, estimator, method, num_fs_wanted=10):
        """
        :param estimator
        :param method: 'rfe', 'lasso', and 'pi'
        :param num_fs_wanted: number of significant features want to select
        """

        if method == 'rfe':
            selected_features = rfe(x=self.X, y=self.y, features=self.features,
                                    num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'pi':
            selected_features = pi(x=self.X, y=self.y, features=self.features,
                                   num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'lasso':
            selected_features = lasso(x=self.X, y=self.y, features=self.features,
                                      estimator=estimator)
        else:
            raise ValueError('unknown feature selection method')

        # update feature names
        self.selectedFeatures = selected_features
        # update predictor values
        self.selectedX = np.asarray(self.df[self.selectedFeatures])
