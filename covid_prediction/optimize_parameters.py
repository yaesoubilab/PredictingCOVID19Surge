import pandas as pd

import covid_prediction.cross_validation as CV
from definitions import ROOT_DIR, get_dataset_labels, get_short_outcome


def get_neural_net_best_spec(outcome_name, week, model_spec, noise_coeff, bias_delay,
                             list_of_alphas, feature_selection, if_standardize,
                             cv_fold, if_parallel=False):
    """
    :param outcome_name: (string) 'Maximum hospitalization rate' or 'If hospitalization threshold passed'
    :param week: (int) week when the predictions should be made
    :param model_spec: (ModelSpec) model specifications
    :param noise_coeff: (None or integer)
    :param list_of_alphas: (list) of regularization penalties
    :param feature_selection: (string) feature selection method
    :param if_standardize: (bool) set True to regularize features
    :param cv_fold: (int) number of cross validation folds
    :param if_parallel: (bool) set True to run code in parallel
    :return: (ModelSpec) the optimal model specification based on R2 score
    """

    # read dataset
    label = get_dataset_labels(
        week=week, noise_coeff=noise_coeff, bias_delay=bias_delay)
    df = pd.read_csv('{}/outputs/prediction_datasets/time_to_peak/data-{}.csv'.format(ROOT_DIR, label))

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

    # scoring and outcome for filenames
    if outcome_name == 'Maximum hospitalization rate':
        scoring = None  # use default which is R2 score
        if_outcome_binary = False
    elif outcome_name == 'If hospitalization threshold passed':
        scoring = 'roc_auc'
        if_outcome_binary = True
    else:
        raise ValueError('Invalid outcome to predict.')
    short_outcome = get_short_outcome(outcome_name)

    # find the best specification
    cv = CV.NeuralNetParameterOptimizer(df=df, feature_names=model_spec.features,
                                        outcome_name=outcome_name, if_outcome_binary=if_outcome_binary,
                                        list_of_n_features_wanted=model_spec.listNumOfFeaturesWanted,
                                        list_of_alphas=list_of_alphas,
                                        list_of_n_neurons=model_spec.listNumOfNeurons,
                                        feature_selection_method=feature_selection,
                                        cv_fold=cv_fold,
                                        scoring=scoring,
                                        if_standardize=if_standardize)

    best_spec = cv.find_best_parameters(
        run_in_parallel=if_parallel,
        save_to_file_performance=ROOT_DIR + '/outputs/prediction_summary/neu_net/cv/eval-predicting {}-{}-{}.csv'
            .format(short_outcome, model_spec.name, label),
        save_to_file_features=ROOT_DIR + '/outputs/prediction_summary/neu_net/features/features-predicting {}-{}-{}.csv'
            .format(short_outcome, model_spec.name, label)
    )

    return best_spec


def optimize_and_eval_dec_tree(model_spec, outcome_name, list_of_max_depths, list_of_ccp_alphas,
                               feature_selection, cv_fold, if_parallel=False, shorten_feature_names=None):
    """
    :param model_spec: (ModelSpec) model specifications
    :param outcome_name: (string) outcome to predict 
    :param list_of_max_depths: (list) of maximum depths
    :param list_of_ccp_alphas: (list) of ccp alphas
    :param feature_selection: (string) feature selection method
    :param cv_fold: (int) number of cross validation folds
    :param if_parallel: (bool) set True to run code in parallel
    :param shorten_feature_names: (dictionary) with keys as features names in the dataset and
        values as alternative names to replace the original names with
    :return: (best specification, the final model performance)
    """

    # number of features
    print('Number of features:', len(model_spec.features))

    # read dataset
    df = pd.read_csv('{}/outputs/prediction_datasets/week_into_fall/combined_data.csv'.format(ROOT_DIR))
    # randomize rows (since the dataset is ordered based on the likelihood weights)
    df = df.sample(frac=1, random_state=1)

    # number of rows
    n_rows = df.shape[0]
    df_training = df.head(int(n_rows*0.8))
    df_validation = df.tail(n_rows - int(n_rows*0.8))

    # find the best specification
    cv = CV.DecTreeParameterOptimizer(
        df=df_training,
        feature_names=model_spec.features,
        outcome_name=outcome_name,
        if_outcome_binary=True,
        list_of_n_features_wanted=None,
        list_of_max_depths=list_of_max_depths,
        list_of_ccp_alphas=list_of_ccp_alphas,
        feature_selection_method=feature_selection,
        cv_fold=cv_fold,
        scoring='accuracy')

    best_spec = cv.find_best_parameters(
        run_in_parallel=if_parallel,
        save_to_file_performance=ROOT_DIR + '/outputs/prediction_summary/dec_tree/cv/eval-{}.csv'
            .format(model_spec.name),
        save_to_file_features=ROOT_DIR + '/outputs/prediction_summary/dec_tree/features/features-{}.csv'
            .format(model_spec.name)
    )

    # train the model model and evaluate it on the validation dataset
    final_model = cv.evaluate_tree_on_validation_set(
        df_training=df_training,
        df_validation=df_validation,
        selected_features=best_spec.selectedFeatures,
        y_name=outcome_name,
        max_depth=best_spec.maxDepth,
        ccp_alpha=best_spec.ccpAlpha)

    # save the best tree
    final_model.plot_decision_path(
        file_name=ROOT_DIR + '/outputs/figures/trees/model-{}.png'.format(model_spec.name),
        simple=True, class_names=['Yes', 'No'],
        precision=2, shorten_feature_names=shorten_feature_names)

    return best_spec, final_model.performanceTest
