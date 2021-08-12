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
                               feature_selection_method, list_of_num_fs_wanted,
                               list_of_penalties, list_of_poly_degrees,
                               n_bootstraps=100, if_standardize=True
                               ):
    # preprocess
    data_lr = Dataframe(df=data, features=feature_names, y_name=outcome_name)
    data_lr.preprocess(if_standardize=if_standardize, degree_of_polynomial=poly_degree)

    # find the best combination
    for penalty in list_of_penalties:
        for poly_degree in list_of_poly_degrees:
            for n_fs in list_of_num_fs_wanted:
                print('penalty', penalty, '; poly_degree', poly_degree, '; num features', n_fs)
                # construct model
                if penalty == 'l1':
                    reg = linear_model.Lasso(alpha=alpha)
                elif penalty == 'l2':
                    reg = linear_model.Ridge(alpha=alpha)
                else:
                    reg = linear_model.LinearRegression()



    # feature selection
    data_lr.feature_selection(estimator=estimator, method=feature_selection_method, num_fs_wanted=n_fs)

    # feed in linear regression model
    models = MultiLinearReg(df=data_lr.df, features=data_lr.features, y_name=data_lr.y_name)
    models.run_many(num_bootstraps=n_bootstraps, penalty=penalty)
    models.performancesTest.print(decimal=3)


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
    list_cv_score = [list_of_num_fs_wanted]
    for penalty in list_of_alphas:
        cv_score_per_alpha = []
        for n_fs in list_of_num_fs_wanted:
            # print('alpha:', ALPHA, '; num_features:', NUM_FEATURES)
            # construct model
            model = MLPRegressor(alpha=penalty, hidden_layer_sizes=(n_neurons,),
                                 max_iter=1000, solver='sgd', activation='logistic')
            # feature selection
            data_nn.feature_selection(estimator=model, method=feature_selection_method, num_fs_wanted=n_fs)

            # cross-validation
            stat_cv = Stat.SummaryStat(name='cross-validation score for one alpha-feature combination',
                                       data=cross_val_score(estimator=model, X=data_nn.X, y=data_nn.y.ravel(), cv=cv_fold))
            cv_score_per_alpha.append(stat_cv.get_formatted_mean_and_interval(deci=outcome_deci, interval_type="p"))
        list_cv_score.append(cv_score_per_alpha)

    # format
    string_of_alphas = ['alpha_{}'.format(alpha) for alpha in list_of_alphas]
    string_of_alphas.insert(0, 'num_features')
    cv_df = pd.DataFrame(np.asarray(list_cv_score).T, columns=string_of_alphas)
    print('\n', cv_df)

    if save_to_file:
        cv_df.to_csv('../outputs/cv_score/week{}_cv_score_{}fold.csv'.format(cv_fold, week))
