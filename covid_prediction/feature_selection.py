import pandas as pd

from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV


def get_selected_features_based_on_TF_result(features, mask):
    """ get features with "True" indicators and discard features with "False" indicator
    :param features: a list containing all features
    :param mask: list of true and false indicating whether corresponding feature should be selected
    :return a list of selected features
    """

    # find the index of selected feature names
    selected_feature_names = []
    for name, judge in zip(features, mask):
        if judge:
            selected_feature_names.append(name)

    return selected_feature_names


def rfe(x, y, features, num_wanted, estimator):
    """ recursive feature elimination (backward elimination)
    :param x: values of features
    :param y: value of y
    :param features: list of names of all features
    :param num_wanted: number of features selected
    :param estimator: model
    """

    # reference for logistic regression:
    # https://towardsdatascience.com/a-look-into-feature-importance-in-logistic-regression-models-a4aa970f9b0f

    # backward algorithm
    selector = RFE(estimator, n_features_to_select=num_wanted, step=1)
    # use logistic regression to fit the data, do backward elimination and delete least important feature
    selector = selector.fit(x, y)
    # a list of indicators for each features
    whether_use = selector.support_
    # get selected features based on True/False result
    selected_feature_names = get_selected_features_based_on_TF_result(features=features, mask=whether_use)

    return selected_feature_names


def lasso(x, y, features, estimator):
    """ use the LASSO algorithm to choose features for logistic regression """

    sel = SelectFromModel(estimator)
    sel.fit(x, y)
    lasso_support = sel.get_support()  # selection decision for each indicator
    # get selected features based on T/F result
    selected_feature_names = get_selected_features_based_on_TF_result(features=features, mask=lasso_support)

    return selected_feature_names


def lasso_cv(x, y, features):
    """ use LASSO for linear regression"""
    reg = LassoCV()


def create_feature_importance_table(features, importance_col_name, importance_value):
    """ create a sorted table containing feature name and its importance score
    :param features: list of feature names
    :param importance_col_name: the name of the column which shows the importance value or rank for each feature
    :param importance_value: the importance value or rank for each feature
    :return a sorted DataFrame containing the name of feature and its corresponding importance
    """

    # creat a DataFrame
    d = {'Features': features, importance_col_name: importance_value}
    df = pd.DataFrame(d)
    # sort by descending
    df = df.sort_values(by=[importance_col_name], ascending=False)
    return df


def pi(x, y, features, num_wanted, estimator):
    """ permutation_importance selection method
    :param x: values of features
    :param y: value of y
    :param features: list of names of all features
    :param num_wanted: number of features selected
    :param estimator: model
    """
    estimator.fit(x, y)
    result = permutation_importance(estimator, x, y, n_repeats=10, random_state=0)
    importance_mean = result.importances_mean
    # create a DataFrame
    df = create_feature_importance_table(features, 'Importance', importance_mean)
    # get n_feature_wanted features
    wanted_features = (df['Features'][0: num_wanted]).tolist()
    return wanted_features
