import SimPy.Statistics as Stat
from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *
from definitions import ROOT_DIR

NUM_OF_NEURONS = None
DECIMAL = 4     # decimal place for cross-validation scores
CV = 10          # num of splits for cross validation
week = '86'
list_of_num_features_wanted = [10, 20]  # [10, 20, 30, 40]
list_of_alphas = [0.001, 0.01]  # [0.0001, 0.001, 0.01, 1]

# read dataset
df = pd.read_csv('../outputs/prediction_dataset/data at week {}.0.csv'.format(week))
# randomize rows
df = df.sample(frac=1, random_state=1)
# y_names
y_name_continues = 'Maximum hospitalization rate'
y_name_binary = 'If hospitalization threshold passed'
# feature names
feature_names = df.columns.tolist()
feature_names.remove(y_name_continues)
feature_names.remove(y_name_binary)

# preprocess (standardize)
data = Dataframe(df=df, features=feature_names, y_name=y_name_continues)
data.preprocess(if_standardize=True)

# TODO: after you check all other TODOs, it might make sense to move this block under
#   the Datafram class. You can call it do_cv_neural_net
#   it could take list_n_neurons, list_n_features, and list_alphas.
#   And you can structure the csv file with these columns
#   # features, # neurons, alpha, R2
# CV to choose alpha + number of features
n_neurons = len(feature_names) + 2 if NUM_OF_NEURONS is None else NUM_OF_NEURONS
list_cv_score = [list_of_num_features_wanted]
for alpha in list_of_alphas:
    cv_score_per_alpha = []
    for n_features in list_of_num_features_wanted:
        print('alpha:', alpha, '; num of features:', n_features)
        # model hyper-parameters
        model = MLPRegressor(alpha=alpha, hidden_layer_sizes=(n_neurons, ),
                             max_iter=1000, solver='sgd', activation='logistic')
        # feature selection
        data.feature_selection(estimator=model, method='pi', num_fs_wanted=n_features)
        # cross-validation
        # TODO: you are using data.X and data.y below but this could be problematic because for
        #   the polynomial model you are storing X values in data.poly_X so I made some changes in the
        #   Dataframe class to avoid this issue. Would you please remove?
        cv_data = cross_val_score(estimator=model, X=data.X, y=data.y, cv=CV)
        stat_cv = Stat.SummaryStat(name='cross-validation score for one alpha-feature combination',
                                   data=cv_data)
        cv_score_per_alpha.append(stat_cv.get_formatted_mean_and_interval(deci=DECIMAL, interval_type="p"))
    list_cv_score.append(cv_score_per_alpha)

# format
string_of_alphas = ['alpha_{}'.format(alpha) for alpha in list_of_alphas]
string_of_alphas.insert(0, 'num_features')
cv_df = pd.DataFrame(np.asarray(list_cv_score).T, columns=string_of_alphas)

print('\n', cv_df)
cv_df.to_csv(ROOT_DIR+'/outputs/cv_score/cv_score_{}fold_week{}.csv'.format(CV, week))

