import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from covid_prediction.feature_selection import rfe, lasso, pi


def standardize(x):
    return StandardScaler().fit_transform(x)


class Dataframe:
    def __init__(self, df, features, y_name):
        self.df = df
        self.features = features
        self.y_name = y_name

        self.X = np.asarray(self.df[self.features])
        self.y = np.asarray(self.df[self.y_name])

    def _standardize(self):
        """ standardize feature values """
        self.X = standardize(np.asarray(self.df[self.features]))
        self.y = standardize(np.asarray(self.df[self.y_name]).reshape(-1, 1))

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.features)
        self.df[self.y_name] = self.y

    def _add_polynomial_term(self, degree_of_polynomial):
        """
        :param degree_of_polynomial: The degree of the polynomial features
        """
        poly = PolynomialFeatures(degree_of_polynomial)
        self.X = poly.fit_transform(self.X)                     # updating feature values
        self.features = poly.get_feature_names(self.features)   # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.features)
        self.df[self.y_name] = self.y

    def preprocess(self, standardization=False, degree_of_polynomial=None):
        if standardization:
            self._standardize()
        if degree_of_polynomial is not None:
            self._add_polynomial_term(degree_of_polynomial=degree_of_polynomial)

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
            selected_features = lasso(x=self.X, y=self.y, features=self.features, estimator=estimator)
        else:
            raise ValueError('unknown feature selection method')

        self.features = selected_features
