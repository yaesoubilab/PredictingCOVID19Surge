import pandas as pd

import covid_prediction.cross_validation as CV
from definitions import ROOT_DIR


def get_nue_net_best_performance(week, model_definition,
                                 list_of_alphas, feature_selection, if_standardize,
                                 cv_fold, if_parallel=False):

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/data at week {}.csv'.format(ROOT_DIR, week))

    # use all features if no feature name is provided
    if model_definition.features is None:
        # feature names (all columns are considered)
        model_definition.features = df.columns.tolist()
        model_definition.features.remove('Maximum hospitalization rate')
        model_definition.features.remove('If hospitalization threshold passed')

    # randomize rows
    df = df.sample(frac=1, random_state=1)

    # find the best specification
    cv = CV.NeuNetSepecOptimizer(data=df, feature_names=model_definition.features,
                                 outcome_name='Maximum hospitalization rate',
                                 list_of_n_features_wanted=model_definition.listNumOfFeaturesWanted,
                                 list_of_alphas=list_of_alphas,
                                 list_of_n_neurons=model_definition.listNumOfNeurons,
                                 feature_selection_method=feature_selection,
                                 cv_fold=cv_fold, if_standardize=if_standardize)

    best_spec = cv.find_best_spec(
        run_in_parallel=if_parallel,
        save_to_file=ROOT_DIR + '/outputs/prediction_summary/NN eval-wk {}-model {}.csv'.format(
            week, model_definition.name))
    return best_spec
