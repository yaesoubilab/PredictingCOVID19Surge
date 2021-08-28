import pandas as pd

import covid_prediction.cross_validation as CV
from definitions import ROOT_DIR


def get_neural_net_best_spec(week, model_spec,
                             list_of_alphas, feature_selection, if_standardize,
                             cv_fold, if_parallel=False):
    """
    :param week: (int) week when the predictions should be made
    :param model_spec: (ModelSpec) model specifications
    :param list_of_alphas: (list) of regularization penalties
    :param feature_selection: (string) feature selection method
    :param if_standardize: (bool) set True to regularize features
    :param cv_fold: (int) number of cross validation folds
    :param if_parallel: (bool) set True to run code in parallel
    :return: (ModelSpec) the optimal model specification based on R2 score
    """

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/data at week {}.csv'.format(ROOT_DIR, week))

    # use all features if no feature name is provided
    if model_spec.features is None:
        # feature names (all columns are considered)
        model_spec.features = df.columns.tolist()
        model_spec.features.remove('Maximum hospitalization rate')
        model_spec.features.remove('If hospitalization threshold passed')

    # number of features
    print('Number of features:', len(model_spec.features))
    # randomize rows (since the dataset is ordered based on the likelihood weights)
    df = df.sample(frac=1, random_state=1)

    # find the best specification
    cv = CV.NeuralNetSepecOptimizer(data=df, feature_names=model_spec.features,
                                    outcome_name='Maximum hospitalization rate',
                                    list_of_n_features_wanted=model_spec.listNumOfFeaturesWanted,
                                    list_of_alphas=list_of_alphas,
                                    list_of_n_neurons=model_spec.listNumOfNeurons,
                                    feature_selection_method=feature_selection,
                                    cv_fold=cv_fold,
                                    if_standardize=if_standardize)

    best_spec = cv.find_best_spec(
        run_in_parallel=if_parallel,
        save_to_file_performance=ROOT_DIR + '/outputs/prediction_summary/cv/NN eval-wk {}-model {}.csv'
            .format(week, model_spec.name),
        save_to_file_features=ROOT_DIR + '/outputs/prediction_summary/features/NN features-wk {}-model {}.csv'
            .format(week, model_spec.name)
    )

    return best_spec
