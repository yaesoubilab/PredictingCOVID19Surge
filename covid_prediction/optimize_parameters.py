import pandas as pd

import covid_prediction.cross_validation as CV
from SimPy.InOutFunctions import write_csv
from covid_prediction.prediction_models import DecisionTree
from definitions import ROOT_DIR, get_outcome_label, SCENARIOS, FILL_TREE


class SummaryOfTreePerformance:

    def __init__(self):
        # make prediction at different weeks
        self.rows = [['Model', 'Threshold', 'CV-Formatted PI']] # 'CV-Score', 'CV-PI',
        for key, value in SCENARIOS.items():
            self.rows[0].extend(['Acc-' + value, 'Sen-' + value, 'Spe-' + value])

    def add(self, model_name, hosp_occu_thresholds, best_spec_and_validation_performance, digits):

        for t in hosp_occu_thresholds:

            # best specification and performance
            best_spec, performance = best_spec_and_validation_performance[str(t)]

            # store results
            result = [model_name,
                      t,
                      # best_spec.meanScore,
                      # best_spec.PI,
                      None if best_spec is None else best_spec.get_formatted_mean_and_interval(deci=digits)]
            for p in performance:
                result.extend([
                    round(p.accuracy, digits),
                    None if p.sen is None else round(p.sen, digits),
                    None if p.spe is None else round(p.spe, digits)
                ])

            self.rows.append(result)

    def print(self, file_name):

        # print summary of results
        write_csv(rows=self.rows,
                  file_name=file_name)


def optimize_and_eval_dec_tree(
        model_spec,
        hosp_occu_thresholds,
        weeks_to_predict,
        list_of_ccp_alphas=None,
        optimal_ccp_alpha=None,
        error_tolerance=0.01,
        cv_fold=10,
        if_parallel=False,
        shorten_feature_names=None):
    """
    :param model_spec: (ModelSpec) model specifications
    :param hosp_occu_thresholds: (list) of thresholds for hospital occupancy
    :param weeks_to_predict: (int) number of weeks to predict in the future
    :param list_of_ccp_alphas: (list) of ccp alphas
    :param optimal_ccp_alpha: (float) if None, the optimal value of alpha will be determined, otherwise the provided
        value will be used to train and evaluate the tree
    :param error_tolerance: (float) a pruner tree will be selected if it's accuracy is less
        that the accuracy of the optimal tree by this amount
    :param cv_fold: (int) number of cross validation folds
    :param if_parallel: (bool) set True to run code in parallel
    :param shorten_feature_names: (dictionary) with keys as features names in the dataset and
        values as alternative names to replace the original names with
    :return: (best specification, the final model performance)
    """

    # read training dataset
    df = pd.read_csv(ROOT_DIR+'/outputs/prediction_datasets_{}_weeks/data-training.csv'.format(weeks_to_predict))
    # randomize rows (since the dataset might have some order)
    df_training = df.sample(frac=1, random_state=1)

    # read validation datasets
    validation_dfs = [None] * len(SCENARIOS)
    i = 0
    for key, value in SCENARIOS.items():
        validation_dfs[i] = pd.read_csv(
            ROOT_DIR+'/outputs/prediction_datasets_{}_weeks/data-validating {}.csv'.format(weeks_to_predict, value))
        i += 1

    # for all thresholds
    validation_performance = {}
    for t in hosp_occu_thresholds:

        # if the optimal value of ccp alpha is not provided
        if optimal_ccp_alpha is None:
            # find the best specification
            cv = CV.DecTreeParameterOptimizer(
                df=df_training,
                feature_names=model_spec.features,
                outcome_name=get_outcome_label(threshold=t),
                list_of_ccp_alphas=list_of_ccp_alphas,
                error_tolerance=error_tolerance,
                cv_fold=cv_fold,
                scoring='accuracy')

            best_spec = cv.find_best_parameters(
                run_in_parallel=if_parallel,
                save_to_file_performance=ROOT_DIR
                                         + '/outputs/prediction_summary_{}_weeks/dec_tree/cv/cv-{}-{}.csv'
                                             .format(weeks_to_predict, model_spec.name, t),
                save_to_file_features=ROOT_DIR
                                      + '/outputs/prediction_summary_{}_weeks/dec_tree/features/features-{}-{}.csv'
                                          .format(weeks_to_predict, model_spec.name, t)
            )
            feature_names = best_spec.selectedFeatures
            ccp_alpha = best_spec.ccpAlpha

        else:
            best_spec = None
            ccp_alpha = optimal_ccp_alpha
            feature_names = model_spec.features

        # make a final decision tree model
        model = DecisionTree(df=df_training,
                             feature_names=feature_names,
                             y_name=get_outcome_label(threshold=t))

        # train the model
        model.train(ccp_alpha=ccp_alpha)

        # validate the final model
        model.validate(validation_dfs=validation_dfs)

        # store summary of validation performance
        validation_performance[str(t)] = (best_spec, model.validationPerformanceSummaries)

        # save the best tree
        if optimal_ccp_alpha is None:
            filename = ROOT_DIR + '/outputs/figures/trees_{}_weeks/{}-{}.png'.format(
                weeks_to_predict, model_spec.name, t)
        else:
            filename = ROOT_DIR + '/outputs/figures/trees_{}_weeks/{}-{}-{}.png'.format(
                weeks_to_predict, model_spec.name, t, optimal_ccp_alpha)

        model.plot_decision_path(
            file_name=filename,
            simple=True, class_names=['Yes', 'No'],
            precision=2, shorten_feature_names=shorten_feature_names, filled=FILL_TREE)

    return validation_performance

