import pandas as pd

import covid_prediction.cross_validation as CV
from covid_prediction.prediction_models import DecisionTree
from definitions import ROOT_DIR, get_outcome_label, SCENARIOS, FILL_TREE


def optimize_and_eval_dec_tree(
        model_spec,
        hosp_occu_thresholds,
        list_of_ccp_alphas,
        cv_fold,
        if_parallel=False,
        shorten_feature_names=None):
    """
    :param model_spec: (ModelSpec) model specifications
    :param hosp_occu_thresholds: (list) of thresholds for hospital occupancy
    :param list_of_ccp_alphas: (list) of ccp alphas
    :param cv_fold: (int) number of cross validation folds
    :param if_parallel: (bool) set True to run code in parallel
    :param shorten_feature_names: (dictionary) with keys as features names in the dataset and
        values as alternative names to replace the original names with
    :return: (best specification, the final model performance)
    """

    # read training dataset
    df = pd.read_csv(ROOT_DIR+'/outputs/prediction_datasets/week_into_fall/data-training.csv')
    # randomize rows (since the dataset might have some order)
    df_training = df.sample(frac=1, random_state=1)

    # read validation datasets
    validation_dfs = [None] * len(SCENARIOS)
    i = 0
    for key, value in SCENARIOS.items():
        validation_dfs[i] = pd.read_csv(
            ROOT_DIR+'/outputs/prediction_datasets/week_into_fall/data-validating {}.csv'.format(value))
        i += 1

    # for all thresholds
    validation_performance = {}
    for t in hosp_occu_thresholds:

        # find the best specification
        cv = CV.DecTreeParameterOptimizer(
            df=df_training,
            feature_names=model_spec.features,
            outcome_name=get_outcome_label(threshold=t),
            list_of_ccp_alphas=list_of_ccp_alphas,
            cv_fold=cv_fold,
            scoring='accuracy')

        best_spec = cv.find_best_parameters(
            run_in_parallel=if_parallel,
            save_to_file_performance=ROOT_DIR + '/outputs/prediction_summary/dec_tree/cv/cv-{}-{}.csv'
                .format(model_spec.name, t),
            save_to_file_features=ROOT_DIR + '/outputs/prediction_summary/dec_tree/features/features-{}-{}.csv'
                .format(model_spec.name, t)
        )

        # make a final decision tree model
        model = DecisionTree(df=df_training,
                             feature_names=best_spec.selectedFeatures,
                             y_name=get_outcome_label(threshold=t))
        # train the model
        model.train(ccp_alpha=best_spec.ccpAlpha)

        # validate the final model
        model.validate(validation_dfs=validation_dfs)

        # store summary of validation performance
        validation_performance[str(t)] = (best_spec, model.validationPerformanceSummaries)

        # save the best tree
        model.plot_decision_path(
            file_name=ROOT_DIR + '/outputs/figures/trees/model-{}-{}.png'.format(model_spec.name, t),
            simple=True, class_names=['Yes', 'No'],
            precision=2, shorten_feature_names=shorten_feature_names, filled=FILL_TREE)

    return validation_performance
