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

        # x, df, features after preprocessing
        self.df = self._df
        self.X = self._X
        self.y = self._y.ravel()
        self.features = self._features
        self.addPoly = False

        # selected features and X after feature selection
        self.selectedFeatures = None
        self.selectedX = None

    def _standardize(self):
        """ standardize feature and outcome of interest """
        # TODO: under __init__ I think np.asarray(self.df[self.features]) is already used
        #   to populate self.X and self.y. do we need to use them here again?
        self.X = standardize(np.asarray(self.df[self._features]))
        # TODO: what does .reshape(-1, 1) do here? Do we need it?
        self.y = standardize(np.asarray(self.df[self._yName]).reshape(-1, 1)).ravel()

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self._features)
        self.df[self._yName] = self.y

    def _add_polynomial_term(self, degree_of_polynomial):
        """
        :param degree_of_polynomial: The degree of the polynomial features
        """
        self.addPoly = True
        poly = PolynomialFeatures(degree_of_polynomial)
        self.X = poly.fit_transform(self._X)                     # updating feature values
        self.features = poly.get_feature_names(self._features)   # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.features)
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
        # TODO: I think we can remove the block that is commented out below
        #   since self.X, self.features, self.df takes the correct values
        #   when polynomial terms are used
        # if self.addPoly:
        #     X = self.X
        #     features = self.features
        #     df = self.df
        # else:
        #     X = self._X
        #     features = self.features
        #     df = self.df
        # TODO: this can be removed too, I moved this at the initialization
        y = self.y.ravel()

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
