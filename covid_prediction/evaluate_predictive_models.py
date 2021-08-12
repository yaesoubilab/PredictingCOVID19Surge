from covid_prediction.prediction_models import *
from covid_prediction.pre_process import *


def linear_regression_model_helper(penalty, alpha):
    if penalty == 'l1':
        reg = linear_model.Lasso(alpha=alpha)   # (default) alpha: 1.0
    elif penalty == 'l2':
        reg = linear_model.Ridge(alpha=alpha)   # (default) alpha: 1.0
    else:
        reg = linear_model.LinearRegression()
    return reg


def create_result_df(cv_object_list, column_name_list):
    object_outcome_list = []
    for cv_object in cv_object_list:
        object_outcome_list.append([cv_object.penalty, cv_object.poly_degree, cv_object.n_fs, cv_object.cv_mean_PI])
    df = pd.DataFrame(object_outcome_list, columns=column_name_list)
    return df


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


def evaluate_linear_regression(data, feature_names, outcome_name, cv_fold, outcome_deci, week,
                               feature_selection_method, list_of_num_fs_wanted,  # feature selection
                               list_of_penalties, list_of_poly_degrees,  # penalty, polynomial
                               if_standardize=True, save_to_file=True
                               ):
    if feature_selection_method == 'lasso':
        list_of_num_fs_wanted = ['constant']

    # preprocess
    data_lr = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_lr.preprocess(if_standardize=if_standardize)

    # find the best combination
    cv_object_list = []  # used for storing outcomes
    for penalty in list_of_penalties:
        for poly_degree in list_of_poly_degrees:
            # add polynomial term
            data_lr.preprocess(degree_of_polynomial=poly_degree)
            for n_fs in list_of_num_fs_wanted:
                # print('penalty', penalty, '; poly_degree', poly_degree, '; num_features', n_fs)
                cv_object = LRCVPerformance(penalty=penalty, poly_degree=poly_degree,
                                            f_s_method=feature_selection_method, n_fs=n_fs)
                # construct model
                model = linear_regression_model_helper(penalty=penalty[0], alpha=penalty[1])
                # feature selection
                data_lr.feature_selection(estimator=model, method=feature_selection_method, num_fs_wanted=n_fs)
                # cross-validation
                cv_score_list = cross_val_score(estimator=model,
                                                X=data_lr.selected_X,
                                                y=data_lr.y.ravel(),
                                                cv=cv_fold)
                cv_object.add_cv_performance(cv_score_list=cv_score_list, deci=outcome_deci)
                cv_object_list.append(cv_object)

    # print outcome and save to csv file
    object_outcome_list = []
    for cv_object in cv_object_list:
        object_outcome_list.append([cv_object.penalty, cv_object.poly_degree,
                                    cv_object.f_s_method, cv_object.n_fs, cv_object.cv_mean_PI])
    cv_df = pd.DataFrame(object_outcome_list, columns=['penalty', 'poly_degree',
                                                       'feature_selection_method', 'num_features',
                                                       'performance'])
    print(cv_df)
    if save_to_file:
        cv_df.to_csv('../outputs/cv_score/week{}_linear_reg_{}fold.csv'.format(week, cv_fold))


def evaluate_neural_network(data, feature_names, outcome_name,
                            list_of_alphas,
                            list_of_num_fs_wanted, feature_selection_method,  # feature selection
                            cv_fold, outcome_deci, week,
                            if_standardize=True, num_of_neurons=None, save_to_file=True
                            ):
    # preprocess
    data_nn = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_nn.preprocess(if_standardize=if_standardize)

    # find the best combination of alpha/subset of features
    n_neurons = len(feature_names) + 2 if num_of_neurons is None else num_of_neurons
    cv_object_list = []
    for penalty in list_of_alphas:
        for n_fs in list_of_num_fs_wanted:
            # print('alpha:', ALPHA, '; num_features:', NUM_FEATURES)
            cv_object = NNCVPerformance(alpha=penalty, n_fs=n_fs)
            # construct model
            model = MLPRegressor(alpha=penalty, hidden_layer_sizes=(n_neurons,),
                                 max_iter=1000, solver='sgd', activation='logistic')
            # feature selection
            data_nn.feature_selection(estimator=model, method=feature_selection_method, num_fs_wanted=n_fs)

            # cross-validation
            cv_score_list = cross_val_score(estimator=model,
                                            X=data_nn.selected_X,
                                            y=data_nn.y.ravel(),
                                            cv=cv_fold)
            cv_object.add_cv_performance(cv_score_list=cv_score_list, deci=outcome_deci)
            cv_object_list.append(cv_object)

    # print outcome and save to csv file
    object_outcome_list = []
    for cv_object in cv_object_list:
        object_outcome_list.append([cv_object.alpha, cv_object.n_fs, cv_object.cv_mean_PI])
    cv_df = pd.DataFrame(object_outcome_list, columns=['penalty', 'num_features', 'performance'])
    print(cv_df)
    if save_to_file:
        cv_df.to_csv('../outputs/cv_score/week{}_nn_{}fold.csv'.format(week, cv_fold))


class CVPerformance:
    def __init__(self, n_fs):
        self.n_fs = n_fs

        # initial
        self.cv_score_list = None
        self.cvStat = None
        self.cv_mean_PI = None

    def add_cv_performance(self, cv_score_list, deci):
        self.cv_score_list = cv_score_list
        self.cvStat = Stat.SummaryStat(name='cross-validation score for each parameter combination',
                                       data=cv_score_list)
        self.cv_mean_PI = self.cvStat.get_formatted_mean_and_interval(deci=deci, interval_type="p")


class LRCVPerformance(CVPerformance):
    def __init__(self, penalty, poly_degree, n_fs, f_s_method):
        """ linear regression cross-validation performances """
        super().__init__(n_fs=n_fs)
        self.penalty = penalty
        self.poly_degree = poly_degree
        self.n_fs = n_fs
        self.f_s_method = f_s_method    # feature selection method


class NNCVPerformance(CVPerformance):
    def __init__(self, alpha, n_fs):
        """ neural network cross-validation performances """
        super().__init__(n_fs=n_fs)
        self.alpha = alpha
