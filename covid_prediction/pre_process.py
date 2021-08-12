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

        self.poly_X = None
        self.poly_df = None
        self.poly_features = []
        self.add_poly = False

        self.selected_features = None
        self.selected_X = None

    def _standardize(self):
        """ standardize feature and outcome of interest """
        self.X = standardize(np.asarray(self.df[self.features]))
        self.y = standardize(np.asarray(self.df[self.y_name]).reshape(-1, 1))

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.features)
        self.df[self.y_name] = self.y

    def _add_polynomial_term(self, degree_of_polynomial):
        """
        :param degree_of_polynomial: The degree of the polynomial features
        """
        self.add_poly = True
        poly = PolynomialFeatures(degree_of_polynomial)
        self.poly_X = poly.fit_transform(self.X)                     # updating feature values
        self.poly_features = poly.get_feature_names(self.features)   # updating feature names

        # updating dataframe
        self.poly_df = pd.DataFrame(self.poly_X, columns=self.poly_features)
        self.poly_df[self.y_name] = self.y

    def preprocess(self, if_standardize=False, degree_of_polynomial=None):
        if if_standardize:
            self._standardize()
        if degree_of_polynomial is not None:
            self._add_polynomial_term(degree_of_polynomial=degree_of_polynomial)

    def feature_selection(self, estimator, method, num_fs_wanted=10):
        """
        :param estimator
        :param method: 'rfe', 'lasso', and 'pi'
        :param num_fs_wanted: number of significant features want to select
        """
        if self.add_poly:
            X = self.poly_X
            features = self.poly_features
            df = self.poly_df
        else:
            X = self.X
            features = self.features
            df = self.df
        y = self.y.ravel()

        if method == 'rfe':
            selected_features = rfe(x=X, y=y, features=features, num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'pi':
            selected_features = pi(x=X, y=y, features=features, num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'lasso':
            selected_features = lasso(x=X, y=y, features=features, estimator=estimator)
        else:
            raise ValueError('unknown feature selection method')

        # update feature names
        self.selected_features = selected_features
        # update predictor values
        self.selected_X = np.asarray(df[self.selected_features])
