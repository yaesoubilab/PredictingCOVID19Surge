from covid_prediction.prediction_models import *
from covid_prediction.pre_process import *


def evaluate_logistic(data, feature_names, outcome_name, poly_degree=1, n_bootstraps=100,
                      if_standardize=True, penalty='l2', C=1):
    """
    :param penalty: 'l1','l2', or 'none' (default 'l2')
    """

    # preprocessing
    data_lr = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_lr.preprocess(if_standardize=if_standardize, degree_of_polynomial=poly_degree)

    # feed the model
    models = MultiLogisticReg(df=data, features=feature_names, y_name=outcome_name)
    models.run_many(num_bootstraps=n_bootstraps, penalty=penalty, C=C)
    models.performancesTest.print(decimal=3)


def evaluate_tree(data, feature_names, outcome_name, n_bootstraps=100,
                  if_standardize=True):
    models = MultiDecisionTrees(df=data, features=feature_names, y_name=outcome_name)
    models.run_many(num_bootstraps=n_bootstraps)
    models.performancesTest.print(decimal=3)


def evaluate_linear_regression(data, feature_names, outcome_name,
                               feature_selection_method=None, n_fs=None, estimator=None,    # add feature selection
                               poly_degree=1, n_bootstraps=100, if_standardize=True, penalty='none'
                               ):
    # preprocess
    data_lr = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_lr.preprocess(if_standardize=if_standardize, degree_of_polynomial=poly_degree)

    # feature selection
    if feature_selection_method is not None:
        data_lr.feature_selection(estimator=estimator, method=feature_selection_method, num_fs_wanted=n_fs)

    # feed in linear regression model
    models = MultiLinearReg(df=data_lr.df, features=data_lr.features, y_name=data_lr.y_name)
    models.run_many(num_bootstraps=n_bootstraps, penalty=penalty)
    models.performancesTest.print(decimal=3)


def evaluate_neural_network(data, feature_names, outcome_name, n_bootstraps,
                            if_standardize=True,                                        # preprocessing
                            feature_selection_method=None, n_fs=None, estimator=None,   # if want add feature selection
                            activation='logistic', solver='sgd', alpha=0.0001, max_iter=1000    # hyper-parameters
                            ):
    # preprocess
    data_nn = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_nn.preprocess(if_standardize=if_standardize)

    # feature selection
    if feature_selection_method is not None:
        data_nn.feature_selection(estimator=estimator, method=feature_selection_method, num_fs_wanted=n_fs)

    # feed in neural network model
    models = MultiNNRegression(df=data_nn.df, features=data_nn.features, y_name=data_nn.y_name)
    models.run_many(num_bootstraps=n_bootstraps,
                    activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)
    models.performancesTest.print(decimal=3)
