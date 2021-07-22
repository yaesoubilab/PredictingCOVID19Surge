from covid_prediction.prediction_models import *


def evaluate_logistic(data, feature_names, outcome_name, poly_degree=1, n_bootstraps=100, if_standardize=True):

    models = MultiLogisticReg(df=data, features=feature_names, y_name=outcome_name)
    models.run_many(num_bootstraps=n_bootstraps, degree_of_polynomial=poly_degree, if_standardize=if_standardize)
    models.performancesTest.print(decimal=3)


def evaluate_tree(data, feature_names, outcome_name, n_bootstraps=100, if_standardize=True):
    models = MultiDecisionTrees(df=data, features=feature_names, y_name=outcome_name)
    models.run_many(num_bootstraps=n_bootstraps)
    models.performancesTest.print(decimal=3)





