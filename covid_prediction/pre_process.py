import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from covid_prediction.feature_selection import rfe, lasso, pi


def standardize(x):
    return StandardScaler().fit_transform(x)


class Dataframe:
    def __init__(self, df, features, y_name):

        self._df = df
        self._features = features
        self._yName = y_name
        self._X = np.asarray(self._df[self._features])
        self._y = np.asarray(self._df[self._yName])

        # x, df, features after preprocess
        self.df = self._df
        self.X = self._X        # X after standardization
        self.polyX = self._X    # X after per polynomial
        self.y = self._y.ravel()
        self.features = self._features
        self.addPoly = False

        # selected features and X after feature selection
        self.selectedFeatures = None
        self.selectedX = None

    def _standardize(self):
        """ standardize feature and outcome of interest """
        self.X = standardize(self._X)
        # TODO: what does .reshape(-1, 1) do here? Do we need it?
        #  turn a (1, n) matrix to (n, 1) matrix
        #  it has sth to do with .ravel(). If delete, it raises error
        self.y = standardize(np.asarray(self._y).reshape(-1, 1)).ravel()

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self._features)
        self.df[self._yName] = self.y

    def _add_polynomial_term(self, degree_of_polynomial):
        """
        :param degree_of_polynomial: The degree of the polynomial features
        """
        self.addPoly = True
        poly = PolynomialFeatures(degree_of_polynomial)
        # polynomial is always done after standardization, so I think we should take in X instead of _X
        self.polyX = poly.fit_transform(self.X)                  # updating feature values
        self.features = poly.get_feature_names(self._features)   # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.polyX, columns=self.features)
        self.df[self._yName] = self.y

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

        # still add a judgement here, because NN do not need polynomial, but linear regression need
        X = self.X
        if self.addPoly:
            X = self.polyX

        if method == 'rfe':
            selected_features = rfe(x=X, y=self.y, features=self.features,
                                    num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'pi':
            selected_features = pi(x=X, y=self.y, features=self.features,
                                   num_wanted=num_fs_wanted, estimator=estimator)
        elif method == 'lasso':
            selected_features = lasso(x=X, y=self.y, features=self.features,
                                      estimator=estimator)
        else:
            raise ValueError('unknown feature selection method')

        # update feature names
        self.selectedFeatures = selected_features
        # update predictor values
        self.selectedX = np.asarray(self.df[self.selectedFeatures])
